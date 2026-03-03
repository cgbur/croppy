# Assets

- `demo.png` — combined image used in the project README
- `tui.png` — source screenshot of the TUI
- `preview-overlay.jpg` — source detection overlay preview

## Regenerating demo.png

Rotate the overlay 90 degrees left, then combine with the TUI screenshot:

```bash
magick preview-overlay.jpg -rotate -90 preview-overlay-rotated.jpg
magick \
  \( tui.png -resize x800 \) \
  \( preview-overlay-rotated.jpg -resize x800 \) \
  +smush 20 demo.png
```
