// Copy text to the clipboard, reporting whether it actually landed there. A
// caller must never claim a copy it did not make.
//
// Two paths, because the async Clipboard API is gated on a secure context and
// this dashboard is routinely served from a plain-HTTP LAN address (a gateway on
// 10.x/100.x is not localhost, so `navigator.clipboard` is undefined there).
// document.execCommand("copy") is deprecated but is the only clipboard write
// such an origin has, so it is the fallback rather than an error message.
export async function copyToClipboard(
  text: string,
  clipboard: Pick<Clipboard, "writeText"> | undefined = navigator.clipboard,
): Promise<boolean> {
  if (clipboard) {
    try {
      await clipboard.writeText(text);
      return true;
    } catch {
      // Permission denied or a detached document: try the legacy path.
    }
  }
  return legacyCopy(text);
}

// Copies from an offscreen textarea, restoring whatever the operator had
// selected so a copy does not disturb the page's own selection.
function legacyCopy(text: string): boolean {
  const source = document.createElement("textarea");
  source.value = text;
  source.readOnly = true;
  // Fixed and transparent rather than display:none, which is not selectable.
  source.style.position = "fixed";
  source.style.top = "-1000px";
  source.style.opacity = "0";
  document.body.appendChild(source);

  const selection = document.getSelection();
  const previous = selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null;
  source.select();

  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch {
    copied = false;
  }

  source.remove();
  if (selection && previous) {
    selection.removeAllRanges();
    selection.addRange(previous);
  }
  return copied;
}
