/**
 * The exceptions to "only you", said wherever content is promised to be private.
 *
 * Break-glass exists, so a promise of privacy that leaves it out is a promise the
 * product does not keep. The operator's half is said only where the deployment
 * turned it on.
 */
export function exceptionalAccess(platformBreakGlass: boolean): string {
  const owners =
    "Your organization's owners can still open it without a share, for an investigation; each opening is recorded and shown to all owners, and the person is not told."
  return platformBreakGlass
    ? `${owners} The team that runs this deployment can do the same where required by law, recorded with a reference.`
    : owners
}
