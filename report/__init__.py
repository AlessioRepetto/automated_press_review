"""HTML report generation module for the Analisi News pipeline."""

from report.data_collector import build_report_data, dump_report_data_json
from report.html_renderer import render_html

__all__ = ["build_report_data", "dump_report_data_json", "render_html"]
