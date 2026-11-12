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
gateway probe that network, so Otari refuses these by default:

- private ranges (`10.x.x.x`, `172.16.x.x` to `172.31.x.x`, `192.168.x.x`, and
  IPv6 `fd00::/8` style addresses);
- link-local addresses such as `169.254.169.254`, the cloud metadata address;
- multicast, unspecified and reserved addresses;
- a hostname that does not resolve when Otari checks it.

Loopback (`127.0.0.1`, `localhost`) is allowed by default, for a server running
beside the gateway.

When you save a server, or press **Test connection**, with a private address,
the request is refused with a message like this:

```text
MCP server host '192.168.1.20' resolves to 192.168.1.20 which is in a private range (RFC 1918 / ULA); rejecting to prevent SSRF. Set OTARI_MCP_ALLOW_PRIVATE_HOSTS=true to override.
```

The last sentence names a setting only the person who runs the gateway can
change. On a hosted Otari that is not you, so you need the tunnel route below.

## Choose a route

| Your situation | Route |
| --- | --- |
| You use a hosted Otari, or any Otari that is not on the same network as the server | [Publish the server through an HTTPS tunnel](#route-1-publish-the-server-through-an-https-tunnel) |
| You run Otari yourself, on the same network as the server | [Let your gateway reach private addresses](#route-2-let-your-own-gateway-reach-the-private-address) |

If you can choose, route 2 keeps the server off the internet entirely. It does
cost you a gateway setting that is all or nothing (see below).

## Route 1: publish the server through an HTTPS tunnel

A tunnel gives the private server a public `https://` address without opening a
port on your router. A small program on your network dials out to the tunnel
provider, and the provider forwards requests back down that connection.

**The server is on the public internet once you do this.** Anyone who learns the
address can send it requests. So the server must require a token on every
request, and you should add the token to Otari (see
[Store the token in Otari](#store-the-token-in-otari)). Do not publish a server
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
- Check a bearer token on every request and answer `401` without one.
- Be reachable by the URL you give Otari with no redirect to a different host or
  port.

Some servers do not fit. Home Assistant's built-in MCP server (`/api/mcp`) is
documented to sign users in with OAuth, and its page does not describe a static
bearer token. Otari sends a token you store; it does not run an OAuth sign-in
with the server. So do not assume Home Assistant works with this guide until you
have tried it. The test button in the next section tells you in one click.

## Route 2: let your own gateway reach the private address

If you run Otari yourself on the same network, or on a machine that can reach
the server, you can change the gateway's setting rather than publish the
server. Two settings control it. Set them where you set the rest of the gateway's
configuration (see [Configuration](configuration.md)).

| Setting | Default | What it does |
| --- | --- | --- |
| `OTARI_MCP_ALLOW_LOOPBACK` | `true` | Allows `127.0.0.1` and `localhost`. Set `false`, `0` or `no` to refuse them. |
| `OTARI_MCP_ALLOW_PRIVATE_HOSTS` | `false` | Set `true`, `1` or `yes` to turn the address check off. |

- If the server runs in the same network namespace as the gateway (both
  directly on one machine), you need no change: `http://127.0.0.1:<port>/mcp` is
  accepted. A gateway in a container does not share the host's loopback, so
  `127.0.0.1` there means the container itself. Use an address of the host or of
  the server that the container can reach. That is usually a private address, so
  it needs the next step.
- If it is on another machine, set `OTARI_MCP_ALLOW_PRIVATE_HOSTS=true` and
  restart the gateway. With it on, the loopback setting no longer matters:
  `OTARI_MCP_ALLOW_LOOPBACK=false` does not refuse loopback while the private-host
  setting is `true`, because the address check is skipped altogether.

With the Docker Compose file from the
[Self-hosted quickstart](self-hosted-quickstart.md), adding the variable to `.env`
is not enough: `docker-compose.yml` only passes into the container the variables
it lists, and this one is not among them. Pass it through an override file next
to `docker-compose.yml`, named `docker-compose.override.yml`, which Compose reads
automatically:

```yaml
services:
  otari:
    environment:
      - OTARI_MCP_ALLOW_PRIVATE_HOSTS=true
```

Then run `docker compose up -d` again so the container is recreated with the new
environment. If you use a mounted config file instead, the same setting is
`mcp_allow_private_hosts: true` in it.

Know what that setting does before you turn it on. It is a single switch for the
whole gateway. There is no way to allow one host and still refuse the rest, so
it lifts the address check for every stored MCP server in every workspace, and
for the guardrail endpoints that share the same check. That includes addresses
like the cloud metadata address. It also lets a hostname that fails to resolve
through. Anyone who can name a server URL can then aim the gateway at your network: not
only workspace admins, but also any API key holder in a workspace where
**Let requests name their own MCP servers** is on. New workspaces start with it
off, but a workspace that existed before access lists keeps it on. Check that
switch in every workspace before you turn the setting on, and keep the gateway's
own network rules (a firewall or an egress policy) as the real limit on where it
can connect.

These two settings are shown on the dashboard's settings page for reference but
cannot be changed there. Change them through the environment or the config file.

A literal address in `100.64.0.0/10`, the range Tailscale uses for devices, is
not in a blocked range, and passes the check without the setting. That matters if
your gateway is itself on your tailnet. A MagicDNS name that resolves there has
not been tested here.

## Store the token in Otari

On either route, add the server as usual (**Tools**, then **MCP servers**, then
**Add MCP server**) and put the server's token in **Authorization token**.

- Otari sends it as a bearer token. It is encrypted before it is stored, using
  `OTARI_SECRET_KEY`, and it is never shown again. The server's page shows only
  that a token is set. To change it, enter a new one; to remove it, clear the
  field.
- A token requires an `https://` URL. This holds even for loopback or a private
  address: `http://127.0.0.1:8123/mcp` with a token is refused. So a private
  server that needs a token also needs TLS on the address Otari dials. Whether
  Otari accepts a self-signed certificate there has not been verified for this
  guide.

Use **Test connection** before you save. It connects once, lists the tools, and
tells you what failed if it did.

## Lock it down

A reachable server is not a safe one until you decide who may use it.

1. On the server's page, set **Exposed tools** to a short list of the tools you
   want the model to see, rather than **Every tool the server offers**. A tool
   that is not exposed is out of reach whatever any rule says.
2. Set the **Default policy** to **Deny**, then add rules for the keys or users
   who need it. [Control who can use it](mcp-servers.md#control-who-can-use-it)
   has the steps.
3. Turn off **Let requests name their own MCP servers** unless you need it, so
   requests cannot reach the server with a URL and token of their own.
4. On a server you control, make the token a dedicated one with the narrowest
   rights it can have, so you can revoke it without touching anything else.

## Troubleshooting

| Message or state | Cause and fix |
| --- | --- |
| `... resolves to <address> which is in a private range (RFC 1918 / ULA); rejecting to prevent SSRF` | The URL points at a private address. Use a tunnel, or on your own gateway set `OTARI_MCP_ALLOW_PRIVATE_HOSTS=true`. |
| `... which is link-local`, `... which is loopback`, `... which is multicast`, `... which is in a reserved range` | Same check, different range. Loopback is refused only when `OTARI_MCP_ALLOW_LOOPBACK` is off. |
| `... could not be resolved at validation time; rejecting to avoid DNS-rebinding` | The gateway cannot look up the name. For a tunnel, check the DNS record exists. For a private name, use `OTARI_MCP_ALLOW_PRIVATE_HOSTS=true` or the address itself. |
| `MCP server URL must use https when a credential travels with it` | A token is set and the URL starts with `http://`. Use an `https://` address. |
| Health `unreachable` or `timeout` | The gateway passed the address check but could not connect. Check the tunnel is healthy and the server is running. |
| Health `tls_failed` | The certificate on the address is not one the gateway trusts. |
| Health `unauthorized` or `forbidden` | The server rejected the token. Re-enter it. |
| Health `not_found` | The path is wrong. Use the exact MCP path your server documents. |
| Health `not_mcp` | The address answered, but not as a streamable HTTP MCP server. |
| Health `redirected` | The server redirected to another host or port. Use the final URL. |
| A request fails with `500` after the server was fine | The stored URL is checked again on every request. The host may now resolve to a private address. The reason is in the gateway log. |

The full list of health values is under
[Connection checks](mcp.md#connection-checks) in the MCP reference.
