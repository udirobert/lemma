"""Build the re:AGENT submission deck directly in Google Slides.

Run:  .venv/bin/python scripts/make_submission_deck.py
Requires token.json (see scripts/google_auth_flow.py) and deck_content.py.

Creates the presentation in the authenticated user's Drive and prints the
edit URL. Each slide is sent as its own batchUpdate so one bad request
cannot abort the deck.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from googleapiclient.discovery import build  # noqa: E402

from scripts.deck_content import SLIDES  # noqa: E402
from scripts.google_auth_flow import get_credentials  # noqa: E402

# --- theme ---------------------------------------------------------------
BG = {"red": 0.043, "green": 0.063, "blue": 0.094}  # deep navy #0B1018
FG = {"red": 0.93, "green": 0.94, "blue": 0.96}  # near-white
ACCENT = {"red": 0.31, "green": 0.56, "blue": 0.97}  # electric blue #4F8EF7
MUTED = {"red": 0.62, "green": 0.66, "blue": 0.72}
GREEN = {"red": 0.13, "green": 0.77, "blue": 0.37}
AMBER = {"red": 0.96, "green": 0.62, "blue": 0.04}

SLIDE_W, SLIDE_H = 960.0, 540.0
FONT = "Arial"
MONO = "Courier New"


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def rgb(color: dict) -> dict:
    # fill-style color (SolidFill.color takes rgbColor directly)
    return {"rgbColor": color}


def textcolor(color: dict) -> dict:
    # TextStyle colors need the opaqueColor wrapper
    return {"opaqueColor": {"rgbColor": color}}


def page_bg() -> dict:
    return {
        "updatePageProperties": {
            "objectId": None,  # filled per slide below
            "pageProperties": {
                "pageBackgroundFill": {"solidFill": {"color": rgb(BG)}},
            },
            "fields": "pageBackgroundFill.solidFill.color",
        }
    }


def textbox(
    slide_id: str,
    obj_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
) -> dict:
    return {
        "createShape": {
            "objectId": obj_id,
            "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {
                    "width": {"magnitude": w, "unit": "PT"},
                    "height": {"magnitude": h, "unit": "PT"},
                },
                "transform": {
                    "scaleX": 1,
                    "scaleY": 1,
                    "translateX": x,
                    "translateY": y,
                    "unit": "PT",
                },
            },
        }
    }


def style_text(
    obj_id: str,
    size: float,
    color: dict | None = None,
    bold: bool = False,
    mono: bool = False,
    start: int = 0,
    end: int | None = None,
) -> dict:
    st: dict = {"fontSize": {"magnitude": size, "unit": "PT"}, "bold": bold}
    if color:
        st["foregroundColor"] = textcolor(color)
    if mono:
        st["fontFamily"] = MONO
    rng = (
        {"type": "ALL"}
        if end is None
        else {"type": "FIXED_RANGE", "startIndex": start, "endIndex": end}
    )
    fields = "fontSize,bold"
    if color:
        fields += ",foregroundColor"
    if mono:
        fields += ",fontFamily"
    return {
        "updateTextStyle": {
            "objectId": obj_id,
            "textRange": rng,
            "style": st,
            "fields": fields,
        }
    }


def para(obj_id: str, size: float, space_after: float = 8) -> dict:
    return {
        "updateParagraphStyle": {
            "objectId": obj_id,
            "textRange": {"type": "ALL"},
            "style": {"spaceBelow": {"magnitude": space_after, "unit": "PT"}},
            "fields": "spaceBelow",
        }
    }


def create_slide(title: str) -> tuple[list, str]:
    sid = _id("slide")
    reqs = [
        {
            "createSlide": {
                "objectId": sid,
                # omit insertionIndex → append to end (deck is cleared first)
                "slideLayoutReference": {"predefinedLayout": "BLANK"},
            }
        }
    ]
    return reqs, sid


def slide_header(reqs: list, sid: str, title: str) -> None:
    tid = _id("title")
    reqs.append(textbox(sid, tid, 48, 30, SLIDE_W - 96, 64))
    reqs.append({"insertText": {"objectId": tid, "text": title}})
    reqs.append(style_text(tid, 27, FG, bold=True))
    # accent underline bar
    bar = _id("bar")
    reqs.append(
        {
            "createShape": {
                "objectId": bar,
                "shapeType": "RECTANGLE",
                "elementProperties": {
                    "pageObjectId": sid,
                    "size": {
                        "width": {"magnitude": 90, "unit": "PT"},
                        "height": {"magnitude": 4, "unit": "PT"},
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": 48,
                        "translateY": 84,
                        "unit": "PT",
                    },
                },
            }
        }
    )
    reqs.append(
        {
            "updateShapeProperties": {
                "objectId": bar,
                "shapeProperties": {
                    "shapeBackgroundFill": {"solidFill": {"color": rgb(ACCENT)}},
                    "outline": {"outlineFill": {"solidFill": {"color": rgb(ACCENT)}}},
                },
                "fields": "shapeBackgroundFill,outline",
            }
        }
    )


def build_title(spec: dict) -> list:
    reqs, sid = create_slide("title")
    reqs.append(
        {
            "updatePageProperties": {
                "objectId": sid,
                "pageProperties": {
                    "pageBackgroundFill": {"solidFill": {"color": rgb(BG)}}
                },
                "fields": "pageBackgroundFill.solidFill.color",
            }
        }
    )
    tid = _id("big")
    reqs.append(textbox(sid, tid, 48, 140, SLIDE_W - 96, 110))
    reqs.append({"insertText": {"objectId": tid, "text": spec["title"]}})
    reqs.append(style_text(tid, 66, FG, bold=True))
    sid2 = _id("sub")
    reqs.append(textbox(sid, sid2, 48, 250, SLIDE_W - 96, 180))
    reqs.append({"insertText": {"objectId": sid2, "text": spec["subtitle"]}})
    reqs.append(style_text(sid2, 16, MUTED))
    reqs.append(para(sid2, 16, 10))
    return reqs


def build_bullets(spec: dict) -> list:
    reqs, sid = create_slide("bullets")
    reqs.append(
        {
            "updatePageProperties": {
                "objectId": sid,
                "pageProperties": {
                    "pageBackgroundFill": {"solidFill": {"color": rgb(BG)}}
                },
                "fields": "pageBackgroundFill.solidFill.color",
            }
        }
    )
    slide_header(reqs, sid, spec["title"])
    bid = _id("body")
    text = "\n".join("•  " + b for b in spec["bullets"])
    reqs.append(textbox(sid, bid, 48, 110, SLIDE_W - 96, SLIDE_H - 150))
    reqs.append({"insertText": {"objectId": bid, "text": text}})
    reqs.append(style_text(bid, 17, FG))
    reqs.append(para(bid, 17, 14))
    return reqs


def build_table(spec: dict) -> list:
    reqs, sid = create_slide("table")
    reqs.append(
        {
            "updatePageProperties": {
                "objectId": sid,
                "pageProperties": {
                    "pageBackgroundFill": {"solidFill": {"color": rgb(BG)}}
                },
                "fields": "pageBackgroundFill.solidFill.color",
            }
        }
    )
    slide_header(reqs, sid, spec["title"])
    headers = spec["table"]["headers"]
    rows = spec["table"]["rows"]
    table_id = _id("tbl")
    reqs.append(
        {
            "createTable": {
                "objectId": table_id,
                "elementProperties": {
                    "pageObjectId": sid,
                    "size": {
                        "width": {"magnitude": 864, "unit": "PT"},
                        "height": {"magnitude": 200, "unit": "PT"},
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": 48,
                        "translateY": 120,
                        "unit": "PT",
                    },
                },
                "rows": len(rows) + 1,
                "columns": len(headers),
            }
        }
    )
    # header row
    for c, h in enumerate(headers):
        reqs.append(
            {
                "insertText": {
                    "objectId": table_id,
                    "cellLocation": {"rowIndex": 0, "columnIndex": c},
                    "text": h,
                }
            }
        )
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            reqs.append(
                {
                    "insertText": {
                        "objectId": table_id,
                        "cellLocation": {"rowIndex": r, "columnIndex": c},
                        "text": val,
                    }
                }
            )
    # style every cell's text (per-cell; there is no range text-style request)
    for r in range(len(rows) + 1):
        for c in range(len(headers)):
            reqs.append(
                {
                    "updateTextStyle": {
                        "objectId": table_id,
                        "cellLocation": {"rowIndex": r, "columnIndex": c},
                        "textRange": {"type": "ALL"},
                        "style": {
                            "fontSize": {"magnitude": 12, "unit": "PT"},
                            "fontFamily": FONT,
                            "bold": r == 0,
                            "foregroundColor": textcolor(ACCENT if r == 0 else FG),
                        },
                        "fields": "fontSize,fontFamily,bold,foregroundColor",
                    }
                }
            )
    return reqs


def build_figure(spec: dict) -> list:
    reqs, sid = create_slide("figure")
    reqs.append(
        {
            "updatePageProperties": {
                "objectId": sid,
                "pageProperties": {
                    "pageBackgroundFill": {"solidFill": {"color": rgb(BG)}}
                },
                "fields": "pageBackgroundFill.solidFill.color",
            }
        }
    )
    slide_header(reqs, sid, spec["title"])
    iw, ih = spec["image_dims"]
    max_w, max_h = 620.0, 250.0
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    x = (SLIDE_W - w) / 2
    reqs.append(
        {
            "createImage": {
                "url": spec["image"],
                "elementProperties": {
                    "pageObjectId": sid,
                    "size": {
                        "width": {"magnitude": w, "unit": "PT"},
                        "height": {"magnitude": h, "unit": "PT"},
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": x,
                        "translateY": 110,
                        "unit": "PT",
                    },
                },
            }
        }
    )
    cid = _id("cap")
    reqs.append(textbox(sid, cid, 88, 380, SLIDE_W - 176, 130))
    reqs.append({"insertText": {"objectId": cid, "text": spec["caption"]}})
    reqs.append(style_text(cid, 12.5, MUTED))
    return reqs


def build_links(spec: dict) -> list:
    reqs, sid = create_slide("links")
    reqs.append(
        {
            "updatePageProperties": {
                "objectId": sid,
                "pageProperties": {
                    "pageBackgroundFill": {"solidFill": {"color": rgb(BG)}}
                },
                "fields": "pageBackgroundFill.solidFill.color",
            }
        }
    )
    slide_header(reqs, sid, spec["title"])
    for i, (label, url) in enumerate(spec["links"]):
        lid = _id(f"link{i}")
        y = 120 + i * 62
        reqs.append(textbox(sid, lid, 48, y, SLIDE_W - 96, 54))
        reqs.append({"insertText": {"objectId": lid, "text": f"{label}\n{url}"}})
        reqs.append(style_text(lid, 15, FG, bold=True, end=len(label)))
        reqs.append(style_text(lid, 12, ACCENT, mono=True, start=len(label) + 1))
        reqs.append(
            {
                "updateTextStyle": {
                    "objectId": lid,
                    "textRange": {
                        "type": "FIXED_RANGE",
                        "startIndex": len(label) + 1,
                        "endIndex": len(label) + 1 + len(url),
                    },
                    "style": {"link": {"url": url}},
                    "fields": "link",
                }
            }
        )
    return reqs


BUILDERS = {
    "title": build_title,
    "bullets": build_bullets,
    "table": build_table,
    "figure": build_figure,
    "links": build_links,
}


def main() -> int:
    creds = get_credentials()
    slides_api = build("slides", "v1", credentials=creds)

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pid",
        help="build into an existing presentation instead of creating a new one",
    )
    args = ap.parse_args()

    if args.pid:
        pid = args.pid
        # clear all existing slides in the reused deck first
        pres = slides_api.presentations().get(presentationId=pid).execute()
        ids = [s["objectId"] for s in pres.get("slides", [])]
        if ids:
            slides_api.presentations().batchUpdate(
                presentationId=pid,
                body={"requests": [{"deleteObject": {"objectId": i}} for i in ids]},
            ).execute()
        auto_slide_id = None
        print(f"Reusing presentation (cleared {len(ids)} slides): {pid}")
    else:
        pres = (
            slides_api.presentations()
            .create(body={"title": "Lemma — re:AGENT submission"})
            .execute()
        )
        pid = pres["presentationId"]
        # remember the one auto-created blank slide so we can remove exactly
        # that slide at the end
        auto_slide_id = pres["slides"][0]["objectId"]
        print("Created presentation")
    url = f"https://docs.google.com/presentation/d/{pid}/edit"

    for i, spec in enumerate(SLIDES):
        try:
            reqs = BUILDERS[spec["kind"]](spec)
            slides_api.presentations().batchUpdate(
                presentationId=pid, body={"requests": reqs}
            ).execute()
            print(f"  slide {i + 1}/{len(SLIDES)} OK — {spec['title'][:60]}")
        except Exception as exc:
            print(f"  slide {i + 1}/{len(SLIDES)} FAILED — {exc}")

    # delete only the auto-generated blank slide from a fresh create
    if auto_slide_id:
        try:
            slides_api.presentations().batchUpdate(
                presentationId=pid,
                body={"requests": [{"deleteObject": {"objectId": auto_slide_id}}]},
            ).execute()
        except Exception as exc:
            print(f"  (could not remove blank slide: {exc})")

    print(f"\nDONE → {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
