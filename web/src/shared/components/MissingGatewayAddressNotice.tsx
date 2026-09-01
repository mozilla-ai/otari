// What stands where a request snippet would have been, when the deployment has
// not said which gateway to send requests to (otari#823).
//
// Shared rather than written twice for the same reason the snippet builders in
// `shared/helpers/requestSnippets.ts` are: the Keys page's one-time reveal and
// the setup guide both hand out a key, and an operator who reads one and then
// the other must not be told two different things about the same deployment.
// This is the copy that replaces those builders' output, so it belongs beside
// them rather than in either feature.
export function MissingGatewayAddressNotice() {
  return (
    <p className="text-caption">
      This deployment has not published the gateway address to send requests to,
      so there is no example to show here. Ask whoever runs it for the base URL,
      then call <code>/v1/chat/completions</code> with this key in an{" "}
      <code>Otari-Key</code> header.
    </p>
  )
}
