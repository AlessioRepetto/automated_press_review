# Report module - Automated Press Review

Standalone module that generates the HTML page **"La Parola Data"**.

## Structure

```
report/
├── run_report.py          # entry point CLI (standalone)
├── config.py              # UI palette, italian labes, font fallback
├── data_collector.py      # PipelineOutput -> report_data dict + JSON dump
├── image_builder.py       # PNG -> base64 data URI
├── html_renderer.py       # Jinja2 render
└── templates/
    ├── template.html
    └── style.css

scripts/
└── pipeline.py            # NUOVO file additivo: run_pipeline()
```

The module is **self-contained**: it uses what was already in `scripts/` for the notebook version.
The only addition is `scripts/pipeline.py`, which incapsulates in a function the notebook's
logic, without changing its behaviour.

## Execution

From the project root:

```bash
python -m report.run_report
```

Output (in `artifacts/`):

```
la_parola_data_YYYYMMDD_HHMM.html      # Single HTML file
la_parola_data_YYYYMMDD_HHMM.json      # dump report_data for debug/replay
```

### Options

```bash
python -m report.run_report --output-dir /altro/path
python -m report.run_report -v          # DEBUG logging
```

## PDF Print

The template includes `@media print` with `@page { size: A4; margin: 2cm; }` and
`page-break-before` in the main sections. From the browser → **Print → Save
as PDF** to obtain a professionally paginated PDF, a section for each page
pagina.

## Rendering features

- **Standalone HTML**: all images are base64 embedded
  (no PNG on disk, no external dependencies). The HTML is movable
  as a single file.
- **CSS inline** in `<head>`: no file `.css` to add.
- **JSON twin**: same content of the dict that feeds the
  template. Replay possible without re-executing the pipeline
  (useful for iterations on template/CSS without paying a price in tokens expended).

## Palette

Chromatic decisions (configurable in `config.py`):

| Variable | Value | Use |
|---|---|---|
| `COLOR_HEADING` | `#355C7D` | all titles (H1/H2/H3) |
| `COLOR_PRIMARY` | `#2E6C80` | accents, rulers, meta headers |
| `COLOR_SECONDARY` | `#6A9A86` | secondary accents (border recap, sub-section rules) |
| `COLOR_TEXT` | `#000000` | text body |

## Language

All static texts are in italian (section titles, labels, footer) and
configurables with the dict `LABELS` in `config.py`.

## Extensibility

- **Changing the sub-title**: `config.SUBTITLE_REPORT`
- **Changing label / word "Generato il"**: `config.LABELS`
- **Changing date format in filename**: `config.STAMP_FORMAT`
- **Adding a section**: new key in `build_report_data`,
  new block `{% if %}` in the template, new block in the CSS optional.
