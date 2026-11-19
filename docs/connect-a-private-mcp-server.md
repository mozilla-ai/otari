---
icon: server
---

# Connect an MCP server on a private network

This guide is for an MCP server that lives on a network the public internet
cannot reach: a home lab, an office LAN, a VPN. It shows two ways to connect it
to Otari and how to lock it down afterward. For the basic steps of adding a
server, see [MCP servers](mcp-servers.md).

## Why Otari refuses a private address

Otari connects to a stored MCP server from the gateway process, not from your
browser or your laptop. Before it does, it checks where the URL points. A URL
that names a server inside someone's network is how an attacker would make a
gateway probe that network, or the gateway's own machine, so Otari refuses:

- loopback (`127.0.0.1`, `::1`, and a name such as `localhost` that resolves
  to one);
- private ranges (`10.x.x.x`, `172.16.x.x` to `172.31.x.x`, `192.168.x.x`, and
  IPv6 `fd00::/8` style addresses);
- link-local addresses such as `169.254.169.254`, the cloud metadata address;
- multicast, unspecified and reserved addresses;
- any other address outside the public address space, such as `100.64.0.0/10`,
  the range Tailscale and carrier-grade NAT use for devices;
- a hostname that does not resolve when Otari checks it;
- a name that only means something on the gateway's own network: a single
  label such as `gateway`, or one ending in `.local`, `.internal` or
  `.localhost`;
- an address written as a name, such as `127.1` or `0177.0.0.1`.

A workspace, an organization or an API key holder chooses an MCP server's
address, not the person running the gateway. Otari checks the address when you
save the server and again on every connection, and connects to the address it
checked, so a name that is re-pointed after it was saved still reaches nothing
internal. The one exception is a host the operator lists by name in
`mcp_private_hosts` (see [route 2](#route-2-list-the-host-on-your-own-gateway)).

When you save a server, or press **Test connection**, with a private address,
the request is refused with a message like this:

```text
MCP server host '192.168.1.20' resolves to 192.168.1.20 which is in a private range (RFC 1918 / ULA); rejecting to prevent SSRF. The MCP server must be on a public address.
```

## Choose a route

| Your situation | Route |
| --- | --- |
| You use a hosted Otari, or any Otari shared with people you do not fully trust | [Publish the server through an HTTPS tunnel](#route-1-publish-the-server-through-an-https-tunnel) |
| You run Otari yourself, for yourself or a trusted team, on a network that can reach the server | [List the host on your own gateway](#route-2-list-the-host-on-your-own-gateway) |

## Route 1: publish the server through an HTTPS tunnel

A tunnel gives the private server a public `https://` address without opening a
port on your router. A small program on your network dials out to the tunnel
provider, and the provider forwards requests back down that connection.

**The server is on the public internet once you do this.** Anyone who learns the
address can send it requests. So the server must require a token on every
request, and you should add the credentials to Otari (see
[Store the credentials in Otari](#store-the-credentials-in-otari)). Do not publish a server
that answers without one.

The two examples below are third-party products. They have their own accounts,
prices, limits and terms, and their steps change. Check their current
documentation before you rely on this page.

### Example: Cloudflare Tunnel

Needs a domain that is already added to a Cloudflare account. In the Cloudflare
dashboard:

1. Go to **Networking**, then **Tunnels**, and select **Create a tunnel**.
2. Name it, then run the install command the dashboard shows for your operating
   system on a machine that can reach the MCP server. The dashboard generates
   that command for you, so copy it from there.
3. Open the tunnel, go to **Routes**, and select **Add route**, then **Published
   application**.
4. Pick a subdomain and domain, for example `mcp.example.com`. For **Service
   URL**, enter where the server listens, for example `http://localhost:8123`.
5. Save, and check the tunnel shows as healthy.

Run the tunnel program on the same machine as the MCP server, so the hop from
the tunnel to the server never leaves that machine. The bearer token travels
over that hop, and a plain `http://` Service URL pointing at another machine
sends it across your LAN unencrypted, where anyone on it could copy it and use it
against the public address. If the server has to be on another machine, give it
an `https://` Service URL with a certificate the tunnel program can verify.

Cloudflare's guide is
[Create a remotely-managed tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/).
The MCP URL you give Otari is the public one, such as
`https://mcp.example.com/mcp`, plus whatever path your server uses.

### Example: Tailscale Funnel

Funnel publishes a service from a machine on your tailnet at a
`https://<device>.<tailnet>.ts.net` address. It needs MagicDNS and HTTPS turned
on for the tailnet, and a `funnel` attribute in your access policy. Funnel can
only listen on ports 443, 8443 and 10000. The simplest form, from Tailscale's
documentation, is:

```bash
tailscale funnel 8123
```

where `8123` is the local port of the MCP server. That form runs in the
foreground: closing the terminal stops the Funnel, and after a reboot or a
Tailscale restart you start it again. To keep it running, add `--bg`
(`tailscale funnel --bg 8123`). To stop a background Funnel, repeat the original
command with `off` at the end, as Tailscale's page describes.

Tailscale's page is
[Tailscale Funnel](https://tailscale.com/kb/1223/funnel); read it for the
prerequisites and the flags.

### What the server must do

- Speak MCP over streamable HTTP. Otari does not use the older SSE transport for
  stored servers.
- Ask for credentials on every request and answer `401` without them, whichever
  way you give Otari the credentials: a token you paste in, or a sign-in (see
  [Store the credentials in Otari](#store-the-credentials-in-otari)).
- Be reachable by the URL you give Otari with no redirect to a different host or
  port.

Some servers only offer a sign-in. Home Assistant's built-in MCP server
(`/api/mcp`) is documented to sign users in with OAuth, and its page does not
describe a static bearer token. Otari can sign in to an MCP server with OAuth, so
this may work, but it has not been tried with Home Assistant here. Whether Home
Assistant's sign-in satisfies what Otari asks of a server (see
[Connect your account to an MCP server](connect-your-account-to-an-mcp-server.md))
is unconfirmed. Try it with **Test connection** or a sign-in before you rely on it.

## Route 2: list the host on your own gateway

If you run Otari yourself, you can name the private hosts it may reach in
`mcp_private_hosts`. Each entry is an exact host, a name or an IP address, with
an optional port:

```bash
OTARI_MCP_PRIVATE_HOSTS=mcp-docs:8000,homeassistant.lan,192.168.1.20:8123
```

In a mounted config file, the same setting is a list:

```yaml
mcp_private_hosts:
  - mcp-docs:8000
  - homeassistant.lan
  - 192.168.1.20:8123
```

How it matches:

- An entry matches only the host written in the server's URL, exactly. There
  are no wildcards, ranges or suffixes: `homeassistant.lan` does not admit
  `other.lan`, and `192.168.1.20` does not admit `192.168.1.21`.
- An entry with a port matches only that port. Without one, any port matches.
  A URL with no port uses 80 for `http` and 443 for `https`.
- A listed host may resolve to loopback or a private address. Otari still
  connects to the address it checked on every request, follows no redirect
  off the server's origin, and refuses link-local addresses (the cloud metadata
  service among them), other metadata addresses and `0.0.0.0` even for a
  listed host.
- A server's sign-in addresses go through the same list, so a sign-in service
  on a private host needs its own entry.
- Guardrail endpoints never read this list.

**Every tenant who can name an MCP server can reach a listed host.** That
includes workspace admins and, in a workspace where **Allow servers outside
this list** is on, any API key holder. So list only hosts you would let all of
them reach, and use this on a single-tenant or trusted deployment, not a shared
one. It is empty by default, it is read only from the environment or the
config file, and the dashboard cannot change it.

With the Docker Compose file from the
[Self-hosted quickstart](self-hosted-quickstart.md), `docker-compose.yml` only
passes into the container the variables it lists, so pass this one through a
`docker-compose.override.yml` next to it, then run `docker compose up -d`
again:

```yaml
services:
  otari:
    environment:
      - OTARI_MCP_PRIVATE_HOSTS=mcp-docs:8000
```

A gateway in a container does not share the host's loopback, so `127.0.0.1`
there means the container itself. Use a name or address the container can
reach, such as the Compose service name of the MCP server.

## Store the credentials in Otari

On either route, add the server as usual (**Tools**, then **MCP servers**, then
**Add MCP server**; [MCP servers](mcp-servers.md) has the steps). **Sign-in mode**
decides what you store:

- **Shared account** with a token: put the server's token in **Authorization
  token**. Otari sends it as a bearer token. It is encrypted before it is stored,
  using `OTARI_SECRET_KEY`, and it is never shown again. The server's page shows
  only that a token is set. To change it, enter a new one; to remove it, clear
  the field.
- **Shared account** with a sign-in: leave the token blank, add the server, and
  sign in for everyone on its page
  ([Set up a Shared server's account](mcp-servers.md#set-up-a-shared-servers-account)).
- **Personal accounts**: each member signs in with their own account
  ([Connect your account to an MCP server](connect-your-account-to-an-mcp-server.md)).

A sign-in needs `public_base_url` and `OTARI_SECRET_KEY` set on the deployment,
or adding the server is refused. See
[Before you start](connect-your-account-to-an-mcp-server.md#before-you-start)
there; it is not repeated here.

What that means on a private network:

- A token requires an `https://` URL. A personal server needs `https://` too.
  A tunnel's public address already is one; a listed private host needs a
  certificate the gateway trusts.
- The addresses a sign-in discovers (the server's authorization, token,
  registration and revocation endpoints) must be `https://` and pass the same
  address check as the server's own URL. If your server's sign-in lives at a
  private address, it is refused the same way: publish it too, or list its
  host in `mcp_private_hosts`. A full sign-in through a tunnel or a listed host
  has not been run for this guide.

Use **Test connection** in the add dialog before you save. It connects once,
lists the tools, and tells you what failed if it did. A **Personal accounts**
server has no token to test with, so the dialog offers no test; check it from
its page once it is saved.

## Lock it down

A reachable server is not a safe one until you decide who may use it.

1. On the server's page, set **Exposed tools** to a short list of the tools you
   want the model to see, rather than **Every tool the server offers**. A tool
   that is not exposed is out of reach whatever any rule says.
2. Set the **Default policy** to **Deny**, then add rules for the keys or users
   who need it. [Control who can use it](mcp-servers.md#control-who-can-use-it)
   has the steps.
3. Choose **Availability** with care. **Every request** adds the server to every
   model request made with a workspace key that holds the `mcp:execute` scope,
   so a coding agent reaches it without naming it. A key without that scope does
   not get it. If an attached server stops answering (a timeout, or it cannot be
   reached), Otari leaves it out of requests for 60 seconds and then tries it
   again. See [Attached to every request](mcp.md#attached-to-every-request).
4. Turn off **Allow servers outside this list** unless you need it, so
   requests cannot reach a server with a URL and token of their own.
5. If you rely on **Read tools** to keep callers to reads, set them yourself.
   Otari takes the labels from the server's first tool listing and does not
   change them afterward, so a server you do not trust to describe its own tools
   should not be left to label them. See [Read tools](mcp.md#read-tools).
6. On a server you control, make the token a dedicated one with the narrowest
   rights it can have, so you can revoke it without touching anything else.

## Troubleshooting

| Message or state | Cause and fix |
| --- | --- |
| `... resolves to <address> which is in a private range (RFC 1918 / ULA); rejecting to prevent SSRF` | The URL points at a private address. Publish the server through a tunnel, or on your own gateway list the host in `mcp_private_hosts`. |
| `... which is refused even for a host in mcp_private_hosts` | The listed host resolves to a link-local, metadata or unspecified address, which no entry admits. |
| `... which is link-local`, `... which is loopback`, `... which is multicast`, `... which is in a reserved range`, `... which is outside the public address space` | Same check, different range. A Tailscale `100.x.x.x` address lands in the last one. |
| `... is a local name` or `... is an address written as a name` | The host is a single-label or `.local` name, or an address in a short form. Use the server's public name, or list the host exactly in `mcp_private_hosts`. |
| `... ends with a dot; write the host without it` | The URL's host is written with a trailing dot (`localhost.`). Remove it. |
| Health `tls_failed` on a listed private host | The gateway trusts only the default certificate store, not `SSL_CERT_FILE` or `SSL_CERT_DIR`, so a certificate from a private CA is refused. |
| `... could not be resolved at validation time; rejecting to avoid DNS-rebinding` | The gateway cannot look up the name. For a tunnel, check the DNS record exists. |
| `MCP server URL must use https when a credential travels with it` | A token is set and the URL starts with `http://`. Use an `https://` address. |
| `Signing in to an MCP server needs public_base_url set ...` | A sign-in needs the deployment's `public_base_url`. Set it, as [Before you start](connect-your-account-to-an-mcp-server.md#before-you-start) says. |
| `The authorization endpoint URL failed its safety check` (or `token endpoint`, `registration endpoint`, `revocation endpoint`) | A sign-in address the server published is not `https://` or is on a private address. Fix the server's sign-in address. The gateway log says which check failed. |
| Health `unsafe_url` | The stored URL failed the address check, before any connection was tried. |
| Health `connection_failed` | Among other causes, the name resolved to a non-public address when the gateway connected, though it was public when checked. Check the server's DNS record. |
| Health `unreachable` or `timeout` | The gateway passed the address check but could not connect. Check the tunnel is healthy and the server is running. On a server set to **Every request**, these also make requests skip it for 60 seconds. |
| Health `tls_failed` | The certificate on the address is not one the gateway trusts. |
| Health `unauthorized` or `forbidden` | The server rejected the credential. Re-enter the token, or sign in again. |
| Health `server_error` or `http_error` | The server answered with a 5xx or another unexpected status. Look at the server's own log. |
| Health `not_found` | The path is wrong. Use the exact MCP path your server documents. |
| Health `not_mcp` | The address answered, but not as a streamable HTTP MCP server. |
| Health `redirected` | The server redirected to another host or port. Use the final URL. |
| A request fails with `500` after the server was fine | The stored URL is checked again on every request. The host may now resolve to a private address. The reason is in the gateway log. |

The full list of health values is under
[Connection checks](mcp.md#connection-checks) in the MCP reference.
