"""Check trace content sealing against a real AWS KMS, in an account of your choosing.

Opt-in and never run in CI: it makes real AWS calls. Point it at a development
account, never production:

    uv run python scripts/check_trace_content_kms.py --profile <profile> --region <region> --account <account id>

It refuses to run unless the profile resolves to ``--account``. It creates two
throwaway keys and an alias, runs otari's own sealing code against them (round
trip, KMS refusing another session or workspace, rotation, an alias moved to a
new key, a listed key that is gone, latency and the batch deadline across many
cold sessions, break-glass calling KMS afresh), then deletes the alias and
schedules both keys for deletion in seven days, whatever happened. It prints
results only, never a key, a data key, a ciphertext or a credential.

The identity needs kms:CreateKey, CreateAlias, UpdateAlias, DeleteAlias,
GenerateDataKey, Decrypt, DescribeKey, EnableKeyRotation, RotateKeyOnDemand and
ScheduleKeyDeletion.
"""

import argparse
import asyncio
import base64
import statistics
import sys
import time
import uuid
from typing import Any

import boto3
from botocore.exceptions import ClientError

from gateway.adapters.content_key_adapter import (
    KMS_WRAPPER,
    SECRET_KEY_WRAPPER,
    AwsKmsContentKeys,
    ContentKeyring,
    SecretKeyContentKeys,
)
from gateway.services.traces._content import SealItem, SpanContent

CONTENT = {"command": "pytest -x"}
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)


def keyring(client: Any, key_id: str, previous: tuple[str, ...] = ()) -> ContentKeyring:
    return ContentKeyring(
        KMS_WRAPPER,
        {SECRET_KEY_WRAPPER: SecretKeyContentKeys(), KMS_WRAPPER: AwsKmsContentKeys(key_id, previous, client=client)},
    )


def wrapped_of(sealed: str) -> bytes:
    part = sealed.split(".")[2]
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


async def run(kms: Any, key_a: str, key_b: str, alias: str) -> None:
    ws, other_ws = uuid.uuid4(), uuid.uuid4()
    session, other_session = "s" + uuid.uuid4().hex[:16], "s" + uuid.uuid4().hex[:16]
    context = {"otari:purpose": "trace-content", "otari:workspace": str(ws), "otari:session": session}

    # 1. Round trip through real KMS.
    sealed = await SpanContent(keyring(kms, alias)).seal(CONTENT, workspace_id=ws, trace_id=session, span_id="t1")
    check("seal through real KMS", sealed is not None and sealed.startswith("otc1.kms."))
    assert sealed is not None
    opened = await SpanContent(keyring(kms, alias)).open(sealed, workspace_id=ws, trace_id=session, span_id="t1")
    check("open through real KMS", opened == CONTENT)

    # 2. KMS itself refuses the data key for another session or workspace.
    for label, wrong in (
        ("another session", {**context, "otari:session": other_session}),
        ("another workspace", {**context, "otari:workspace": str(other_ws)}),
        ("no session at all", {k: v for k, v in context.items() if k != "otari:session"}),
    ):
        try:
            kms.decrypt(CiphertextBlob=wrapped_of(sealed), EncryptionContext=wrong)
            check(f"KMS refuses the key for {label}", False, "decrypted")
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            check(f"KMS refuses the key for {label}", code == "InvalidCiphertextException", code)
    moved = await SpanContent(keyring(kms, alias)).open(sealed, workspace_id=ws, trace_id=other_session, span_id="t1")
    check("otari reads a value moved to another session as empty", moved == {})

    # 3. Automatic rotation leaves content readable.
    kms.enable_key_rotation(KeyId=key_a)
    try:
        kms.rotate_key_on_demand(KeyId=key_a)
        rotated = "rotated on demand"
    except ClientError as exc:
        rotated = f"on-demand rotation not run ({exc.response['Error']['Code']})"
    after = await SpanContent(keyring(kms, alias)).open(sealed, workspace_id=ws, trace_id=session, span_id="t1")
    check("content opens after rotation", after == CONTENT, rotated)

    # 4. Alias moved to a new key: old content opens while the old key is listed, not otherwise.
    kms.update_alias(AliasName=alias, TargetKeyId=key_b)
    time.sleep(5)  # alias updates are eventually consistent
    listed = keyring(kms, alias, previous=(key_a,))
    resealed = await SpanContent(listed).seal(CONTENT, workspace_id=ws, trace_id=other_session, span_id="t2")
    assert resealed is not None
    new_key = kms.decrypt(
        CiphertextBlob=wrapped_of(resealed), EncryptionContext={**context, "otari:session": other_session}
    )["KeyId"]
    check("new content is sealed under the new key", new_key.endswith(key_b))
    check(
        "old content opens with the old key listed",
        await SpanContent(listed).open(sealed, workspace_id=ws, trace_id=session, span_id="t1") == CONTENT,
    )
    check(
        "old content reads empty with the old key unlisted",
        await SpanContent(keyring(kms, alias)).open(sealed, workspace_id=ws, trace_id=session, span_id="t1") == {},
    )
    skipping = keyring(kms, alias, previous=(key_a, "alias/otari-does-not-exist"))
    check(
        "a listed key that cannot be described is skipped",
        await SpanContent(skipping).open(sealed, workspace_id=ws, trace_id=session, span_id="t1") == CONTENT,
    )

    # 5. Latency: real GenerateDataKey, and the batch deadline across many cold sessions.
    single: list[float] = []
    for _ in range(10):
        started = time.perf_counter()
        kms.generate_data_key(KeyId=key_b, KeySpec="AES_256", EncryptionContext=context)
        single.append((time.perf_counter() - started) * 1000)
    check(
        "GenerateDataKey latency",
        True,
        f"median {statistics.median(single):.0f} ms, max {max(single):.0f} ms over 10 calls",
    )
    for sessions in (10, 50, 200):
        content = SpanContent(keyring(kms, alias, previous=(key_a,)))
        items = [SealItem(CONTENT, ws, f"s{uuid.uuid4().hex[:16]}", "t1") for _ in range(sessions)]
        started = time.perf_counter()
        out = await content.seal_many(items)
        elapsed = time.perf_counter() - started
        sealed_count = sum(1 for value in out if value is not None)
        check(
            f"one batch across {sessions} cold sessions",
            sealed_count == sessions and elapsed < 1.6,
            f"{sealed_count}/{sessions} sealed in {elapsed * 1000:.0f} ms (budget 1500 ms)",
        )

    # 6. Break-glass opens afresh: a reading with its own keys calls KMS even when the cache is warm.
    warm = SpanContent(keyring(kms, alias, previous=(key_a,)))
    await warm.open(sealed, workspace_id=ws, trace_id=session, span_id="t1")
    calls: list[str] = []
    real_decrypt = kms.decrypt

    def counting_decrypt(**kwargs: Any) -> Any:
        calls.append("Decrypt")
        return real_decrypt(**kwargs)

    kms.decrypt = counting_decrypt
    try:
        await warm.open(sealed, workspace_id=ws, trace_id=session, span_id="t1")
        cached_calls = len(calls)
        await warm.open(sealed, workspace_id=ws, trace_id=session, span_id="t1", keys_opened={})
        fresh_calls = len(calls) - cached_calls
    finally:
        kms.decrypt = real_decrypt
    check(
        "break-glass reading calls KMS even with a warm cache",
        cached_calls == 0 and fresh_calls == 1,
        f"cached read: {cached_calls} Decrypt, break-glass read: {fresh_calls} Decrypt",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, help="The AWS profile to use.")
    parser.add_argument("--region", required=True, help="The region to create the throwaway keys in.")
    parser.add_argument("--account", required=True, help="The account the profile must resolve to, or nothing runs.")
    args = parser.parse_args()
    aws = boto3.Session(profile_name=args.profile, region_name=args.region)
    account = aws.client("sts").get_caller_identity()["Account"]
    if account != args.account:
        print(f"Refusing: the profile resolves to account {account}, not {args.account}")
        return 2
    kms = aws.client("kms")
    key_a = kms.create_key(Description="otari live check A (throwaway)")["KeyMetadata"]["KeyId"]
    key_b = kms.create_key(Description="otari live check B (throwaway)")["KeyMetadata"]["KeyId"]
    alias = f"alias/otari-live-check-{uuid.uuid4().hex[:8]}"
    kms.create_alias(AliasName=alias, TargetKeyId=key_a)
    print(f"Created two throwaway keys and {alias} in {account} ({args.region})", flush=True)
    try:
        asyncio.run(run(kms, key_a, key_b, alias))
    finally:
        try:
            kms.delete_alias(AliasName=alias)
        except ClientError as exc:
            print(f"Could not delete {alias}: {exc.response['Error']['Code']}")
        for key in (key_a, key_b):
            try:
                kms.schedule_key_deletion(KeyId=key, PendingWindowInDays=7)
            except ClientError as exc:
                print(f"Could not schedule deletion of a test key: {exc.response['Error']['Code']}")
        print("Cleanup: alias deleted, both test keys scheduled for deletion in 7 days", flush=True)
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
