"""Build report.html from report.md for browser printToPDF."""

import base64
import re
from pathlib import Path

import markdown

HERE = Path(__file__).parent
FIG = HERE / "figures"
md_text = (HERE / "report.md").read_text()


def img_b64(name):
    return (
        "data:image/png;base64," + base64.b64encode((FIG / name).read_bytes()).decode()
    )


def fig(name, caption, width="80%"):
    return (
        f'<div class="fig"><img src="{img_b64(name)}" style="width:{width}">'
        f'<div class="figcap">{caption}</div></div>'
    )


F1 = fig(
    "fig1_verdict_distribution.png",
    "Figure 1. Per-paper verdict distribution across the audit corpus.",
    "95%",
)
F2 = fig(
    "fig2_test_retest.png",
    "Figure 2. Verdict agreement matrix across the two same-claims pairs "
    "(run A rows vs run B columns; diagonal = agreement).",
    "55%",
)
F3 = fig(
    "fig3_nu_window.png",
    "Figure 3. Same script, same data — the fitted exponent converges to "
    "the claimed (D+1)/2 (dotted lines) as the fit window approaches the "
    "transition. The falsified verdict was a window artifact.",
    "70%",
)

# Inject figures at anchors
md_text = md_text.replace(
    "### 4.3 Test-retest", F1.replace("\n", " ") + "\n\n### 4.3 Test-retest"
)
md_text = md_text.replace(
    "(Figure 3 in the appendix shows",
    F2.replace("\n", " ") + "\n\n(Figure 3 in the appendix shows",
)
md_text = md_text.replace(
    "exactly the kind of failure only re-audits or recorded-trace "
    "inspection can catch.",
    "exactly the kind of failure only re-audits or recorded-trace "
    "inspection can catch.\n\n" + F3.replace("\n", " "),
)

body = markdown.markdown(md_text, extensions=["tables", "smarty"])
# Split title/author block from content: first <h1> stays centered
body = re.sub(r"<h1>(.*?)</h1>", r'<h1 class="title">\1</h1>', body, count=1)

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@page {{ size: letter; margin: 0.8in 0.85in; }}
body {{ font-family: "Times New Roman", Times, serif; font-size: 10.5pt;
  line-height: 1.32; color: #111; max-width: none; }}
h1.title {{ font-size: 15pt; text-align: center; margin: 0 0 8pt; }}
h2 {{ font-size: 11.5pt; margin: 12pt 0 4pt; border-bottom: none; }}
h3 {{ font-size: 10.5pt; margin: 8pt 0 3pt; }}
p {{ margin: 0 0 5pt; text-align: justify; }}
table {{ border-collapse: collapse; margin: 6pt auto; font-size: 9.5pt; }}
th, td {{ border: 0.5pt solid #999; padding: 2.5pt 7pt; text-align: left; }}
th {{ background: #eee; }}
ul, ol {{ margin: 3pt 0 5pt 18pt; padding: 0; }}
li {{ margin-bottom: 2pt; }}
code {{ font-family: "SF Mono", Menlo, monospace; font-size: 9pt; }}
.fig {{ text-align: center; margin: 8pt 0; }}
.figcap {{ font-size: 9pt; font-style: italic; margin-top: 3pt;
  text-align: center; }}
blockquote {{ margin: 4pt 12pt; }}
strong {{ font-weight: bold; }}
</style></head><body>
{body}
</body></html>"""
(HERE / "report.html").write_text(html)
print("wrote report.html")
