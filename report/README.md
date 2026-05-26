# Report module — Analisi News

Modulo standalone che genera l'HTML giornaliero **"La Parola Data"** a partire
dalla pipeline esistente.

## Struttura

```
report/
├── run_report.py          # entry point CLI (standalone)
├── config.py              # palette UI, label italiani, font fallback
├── data_collector.py      # PipelineOutput -> report_data dict + JSON dump
├── image_builder.py       # PNG -> base64 data URI
├── html_renderer.py       # Jinja2 render
└── templates/
    ├── template.html
    └── style.css

scripts/
└── pipeline.py            # NUOVO file additivo: run_pipeline()
```

Il modulo è **autocontenuto**: non modifica niente di esistente in `scripts/`.
L'unica aggiunta è `scripts/pipeline.py`, che incapsula in una funzione la
logica del notebook senza alterarne il comportamento.

## Esecuzione

Dalla project root:

```bash
python -m report.run_report
```

Output (in `artifacts/`):

```
la_parola_data_YYYYMMDD_HHMM.html      # HTML singolo file autoportante
la_parola_data_YYYYMMDD_HHMM.json      # dump report_data per debug/replay
```

### Opzioni

```bash
python -m report.run_report --output-dir /altro/path
python -m report.run_report -v          # DEBUG logging
```

## Stampa PDF

Il template include `@media print` con `@page { size: A4; margin: 2cm; }` e
`page-break-before` sulle sezioni principali. Dal browser → **Stampa → Salva
come PDF** si ottiene un PDF impaginato professionalmente, una sezione per
pagina.

## Caratteristiche del rendering

- **HTML autoportante**: tutte le immagini sono embeddate in base64
  (nessun PNG su disco, nessuna dipendenza esterna). L'HTML è
  spostabile come singolo file.
- **CSS inline** nel `<head>`: nessun file `.css` da accompagnare.
- **Font fallback automatico**: la pipeline usa Arial Windows
  hardcoded. Il modulo monkey-patcha `PIL.ImageFont.truetype` con
  fallback automatico a DejaVu / Liberation / Helvetica per
  funzionare anche su Linux/Mac. Il patch è no-op su Windows con
  Arial presente.
- **JSON gemello**: stesso contenuto del dict che alimenta il
  template. Replay di rendering possibile senza ri-eseguire pipeline
  (utile per iterare su template/CSS senza pagare il costo NLP).

## Palette

Decisioni cromatiche (configurabili in `config.py`):

| Variabile | Valore | Uso |
|---|---|---|
| `COLOR_HEADING` | `#355C7D` | tutti i titoli (H1/H2/H3) |
| `COLOR_PRIMARY` | `#2E6C80` | accenti, righelli, intestazioni meta |
| `COLOR_SECONDARY` | `#6A9A86` | accenti secondari (bordo recap, regola sotto-sezioni) |
| `COLOR_TEXT` | `#000000` | testo body |

La palette dei grafici (`viz_palette.py`) **non** è toccata.

## Lingua

Tutti i testi statici sono in italiano (titoli sezioni, label, footer) e
configurabili nel dict `LABELS` in `config.py`.

## Estensibilità

- **Cambiare il sottotitolo**: `config.SUBTITLE_REPORT`
- **Cambiare label / parola "Generato il"**: `config.LABELS`
- **Cambiare formato data nel filename**: `config.STAMP_FORMAT`
- **Aggiungere una sezione**: nuova chiave in `build_report_data`,
  nuovo blocco `{% if %}` nel template, nuovo blocco CSS opzionale.
