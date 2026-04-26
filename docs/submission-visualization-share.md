# Submission Visualization Sharing Guide

Use this when you need a public, visual link for the DroneCaptureOps system diagram.

## Recommended Free Option: Mermaid Live Editor

Mermaid Live Editor is the official free Mermaid editor and viewer:

<https://mermaid.live/>

Workflow:

1. Open <https://mermaid.live/>.
2. Open `docs/submission-hero-flow.mmd`.
3. Copy the entire Mermaid source.
4. Paste it into the Mermaid Live Editor code panel.
5. Wait for the preview to render.
6. Use the editor's share/copy-link option, or copy the generated URL from the browser bar.
7. Share that URL with judges/team members.

This produces an editable visual link. It is the easiest option for a free shareable visualization.

## Direct SVG Option: Mermaid Ink

Mermaid Ink can render a Mermaid diagram directly as an SVG URL:

<https://mermaid.ink/>

This is useful when you want a direct image link instead of an editor link.

## Generate Both Links Locally

Run this from the repo root:

```bash
python3.11 scripts/generate-mermaid-share-url.py docs/submission-hero-flow.mmd
```

The script prints:

- A Mermaid Live Editor URL: best for editable share links.
- A Mermaid Ink SVG URL: best for embedding or direct visual sharing.

## Which File To Use

Use this file for the shareable one-shot visual:

```text
docs/submission-hero-flow.mmd
```

Use this file for the detailed written diagram appendix:

```text
docs/submission-system-flow.md
```

## Notes

- Mermaid Live links encode the full diagram into the URL, so the link can be long.
- If a platform rejects a long URL, paste `docs/submission-hero-flow.mmd` directly into Mermaid Live Editor and use its built-in export to download SVG/PNG.
- GitHub also renders Mermaid diagrams inside Markdown files, so after pushing, `docs/submission-system-flow.md` can be viewed directly in the repository.
