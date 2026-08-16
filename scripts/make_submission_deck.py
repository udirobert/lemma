"""Build the re:AGENT submission deck directly in Google Slides.

Design system: dark navy, spectrum glass-xylophone accents, Montserrat/
Inter/Roboto Mono, cards, chips, stat panels, full-bleed art with scrims,
and a unifying 5-band spectrum strip on every slide.

Run:  .venv/bin/python scripts/make_submission_deck.py [--pid ID]
Requires token.json (scripts/google_auth_flow.py) and deck_content.py.
Each slide is one batchUpdate; --pid clears and rebuilds an existing deck.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from googleapiclient.discovery import build  # noqa: E402

from scripts.deck_content import SLIDES  # noqa: E402
from scripts.google_auth_flow import get_credentials  # noqa: E402

# --- palette --------------------------------------------------------------
BG = {"red": 0.043, "green": 0.063, "blue": 0.094}  # deep navy #0B1018
BG2 = {"red": 0.075, "green": 0.098, "blue": 0.145}  # raised navy #131925
CARD = {"red": 0.102, "green": 0.129, "blue": 0.184}  # card surface #1A212F
CARD_EDGE = {"red": 0.176, "green": 0.216, "blue": 0.302}  # #2D374D
FG = {"red": 0.945, "green": 0.953, "blue": 0.965}  # near-white #F1F3F6
MUT = {"red": 0.545, "green": 0.588, "blue": 0.651}  # slate #8B96A6
DIM = {"red": 0.392, "green": 0.431, "blue": 0.498}  # #646E7F
WHITE = {"red": 1.0, "green": 1.0, "blue": 1.0}
INK = {"red": 0.063, "green": 0.078, "blue": 0.110}  # dark text on cards

SPEC = {
    "blue": {"red": 0.31, "green": 0.56, "blue": 0.97},  # #4F8EF7
    "magenta": {"red": 0.83, "green": 0.27, "blue": 0.95},  # #D445F2
    "teal": {"red": 0.12, "green": 0.76, "blue": 0.71},  # #1EC2B5
    "gold": {"red": 0.98, "green": 0.71, "blue": 0.20},  # #FAB533
    "violet": {"red": 0.47, "green": 0.40, "blue": 0.93},  # #7866ED
}
SPEC_ORDER = ["blue", "magenta", "teal", "gold", "violet"]

SLIDE_W, SLIDE_H = 960.0, 540.0
FONT_D = "Montserrat"  # display / titles
FONT_B = "Inter"  # body
FONT_M = "Roboto Mono"  # metrics, code, urls

MARGIN = 48.0
CONTENT_W = SLIDE_W - 2 * MARGIN
STRIP_H = 5.0


# --- request helpers -------------------------------------------------------
def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def rgb(color: dict) -> dict:
    return {"rgbColor": color}


def textcolor(color: dict) -> dict:
    return {"opaqueColor": {"rgbColor": color}}


def _size(w: float, h: float) -> dict:
    return {
        "width": {"magnitude": w, "unit": "PT"},
        "height": {"magnitude": h, "unit": "PT"},
    }


def _transform(x: float, y: float) -> dict:
    return {"scaleX": 1, "scaleY": 1, "translateX": x, "translateY": y, "unit": "PT"}


def rect(
    slide_id: str,
    obj_id: str,
    x,
    y,
    w,
    h,
    fill,
    corner_radius=0.0,
    line_color=None,
    line_w=1.0,
):
    shape_type = "ROUND_RECTANGLE" if corner_radius else "RECTANGLE"
    req = {
        "createShape": {
            "objectId": obj_id,
            "shapeType": shape_type,
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": _size(w, h),
                "transform": _transform(x, y),
            },
        }
    }
    props = {"shapeBackgroundFill": {"solidFill": {"color": rgb(fill)}}}
    if line_color:
        props["outline"] = {
            "outlineFill": {"solidFill": {"color": rgb(line_color)}},
            "weight": {"magnitude": line_w, "unit": "PT"},
            "propertyState": "RENDERED",
        }
    return [
        req,
        {
            "updateShapeProperties": {
                "objectId": obj_id,
                "shapeProperties": props,
                "fields": "shapeBackgroundFill,outline,contentAlignment",
            }
        },
    ]


def textbox(slide_id, obj_id, x, y, w, h, content_alignment="TOP_LEFT"):
    return {
        "createShape": {
            "objectId": obj_id,
            "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": _size(w, h),
                "transform": _transform(x, y),
            },
        }
    }


def style_text(obj_id, size, color=None, bold=False, font=FONT_B, start=0, end=None):
    st = {"fontSize": {"magnitude": size, "unit": "PT"}, "bold": bold}
    fields = "fontSize,bold"
    if color:
        st["foregroundColor"] = textcolor(color)
        fields += ",foregroundColor"
    if font:
        st["fontFamily"] = font
        fields += ",fontFamily"
    rng = (
        {"type": "ALL"}
        if end is None
        else {"type": "FIXED_RANGE", "startIndex": start, "endIndex": end}
    )
    return {
        "updateTextStyle": {
            "objectId": obj_id,
            "textRange": rng,
            "style": st,
            "fields": fields,
        }
    }


def para_style(obj_id, space_below=8, alignment=None, line_spacing=None):
    st = {"spaceBelow": {"magnitude": space_below, "unit": "PT"}}
    fields = "spaceBelow"
    if alignment:
        st["alignment"] = alignment
        fields += ",alignment"
    if line_spacing:
        st["lineSpacing"] = line_spacing
        fields += ",lineSpacing"
    return {
        "updateParagraphStyle": {
            "objectId": obj_id,
            "textRange": {"type": "ALL"},
            "style": st,
            "fields": fields,
        }
    }


def create_slide():
    sid = _id("slide")
    return [
        {
            "createSlide": {
                "objectId": sid,
                "slideLayoutReference": {"predefinedLayout": "BLANK"},
            }
        }
    ], sid


def page_bg(sid):
    return {
        "updatePageProperties": {
            "objectId": sid,
            "pageProperties": {"pageBackgroundFill": {"solidFill": {"color": rgb(BG)}}},
            "fields": "pageBackgroundFill.solidFill.color",
        }
    }


def spectrum_strip(sid, y=None):
    """The unifying 5-band accent strip at the bottom of every content slide."""
    reqs = []
    if y is None:
        y = SLIDE_H - STRIP_H
    seg_w = SLIDE_W / len(SPEC_ORDER)
    for i, name in enumerate(SPEC_ORDER):
        oid = _id("strip")
        reqs += rect(sid, oid, i * seg_w, y, seg_w + 0.5, STRIP_H, SPEC[name])
    return reqs


def slide_header(reqs, sid, title, eyebrow=None):
    y = 34.0
    if eyebrow:
        eid = _id("eyebrow")
        reqs.append(textbox(sid, eid, MARGIN, y - 16, CONTENT_W, 22))
        reqs.append({"insertText": {"objectId": eid, "text": eyebrow.upper()}})
        reqs.append(style_text(eid, 10.5, MUT, bold=True, font=FONT_M))
        y += 14
    tid = _id("title")
    reqs.append(textbox(sid, tid, MARGIN, y, CONTENT_W, 58))
    reqs.append({"insertText": {"objectId": tid, "text": title}})
    reqs.append(style_text(tid, 25, FG, bold=True, font=FONT_D))
    # two-tone underline: short spectrum segment
    bar = _id("bar")
    reqs += rect(sid, bar, MARGIN, y + 46, 56, 3.5, SPEC["blue"])
    reqs += rect(sid, _id("bar"), MARGIN + 56, y + 46, 20, 3.5, SPEC["magenta"])
    return y + 46


def full_bleed(reqs, sid, url):
    reqs.append(
        {
            "createImage": {
                "url": url,
                "elementProperties": {
                    "pageObjectId": sid,
                    "size": _size(SLIDE_W, SLIDE_H),
                    "transform": _transform(0, 0),
                },
            }
        }
    )


def scrim(reqs, sid, opacity_from="top"):
    """Dark vertical gradient scrim so text reads over art.

    Slides API has no gradient alpha, so stack translucent navy bands.
    """
    bands = 7
    for i in range(bands):
        frac = i / bands
        # stronger toward the bottom
        alpha = 0.28 + 0.62 * frac
        oid = _id("scrim")
        reqs.append(
            {
                "createShape": {
                    "objectId": oid,
                    "shapeType": "RECTANGLE",
                    "elementProperties": {
                        "pageObjectId": sid,
                        "size": _size(SLIDE_W, SLIDE_H / bands + 0.6),
                        "transform": _transform(0, frac * SLIDE_H),
                    },
                }
            }
        )
        reqs.append(
            {
                "updateShapeProperties": {
                    "objectId": oid,
                    "shapeProperties": {
                        "shapeBackgroundFill": {
                            "solidFill": {"color": rgb(BG), "alpha": alpha}
                        },
                        "outline": {"propertyState": "NOT_RENDERED"},
                    },
                    "fields": "shapeBackgroundFill,outline",
                }
            }
        )


# --- slide builders --------------------------------------------------------
def build_hero(spec):
    reqs, sid = create_slide()
    reqs.append(page_bg(sid))
    full_bleed(reqs, sid, spec["art"])
    scrim(reqs, sid)
    y = 150.0
    kid = _id("kicker")
    reqs.append(textbox(sid, kid, MARGIN, y, CONTENT_W, 26))
    reqs.append({"insertText": {"objectId": kid, "text": spec["kicker"]}})
    reqs.append(style_text(kid, 13, SPEC["teal"], bold=True, font=FONT_M))
    y += 34
    tid = _id("title")
    reqs.append(textbox(sid, tid, MARGIN, y, CONTENT_W, 92))
    reqs.append({"insertText": {"objectId": tid, "text": spec["title"]}})
    reqs.append(style_text(tid, 72, FG, bold=True, font=FONT_D))
    y += 96
    sid2 = _id("sub")
    reqs.append(textbox(sid, sid2, MARGIN, y, CONTENT_W, 70))
    reqs.append({"insertText": {"objectId": sid2, "text": spec["subtitle"]}})
    reqs.append(style_text(sid2, 18, FG))
    reqs.append(para_style(sid2, 4))
    fid = _id("foot")
    reqs.append(textbox(sid, fid, MARGIN, SLIDE_H - 40, CONTENT_W, 24))
    reqs.append({"insertText": {"objectId": fid, "text": spec["footer"]}})
    reqs.append(style_text(fid, 12, MUT, font=FONT_M))
    reqs += spectrum_strip(sid, SLIDE_H - STRIP_H)
    return reqs


def build_bullets(spec):
    reqs, sid = create_slide()
    reqs.append(page_bg(sid))
    slide_header(reqs, sid, spec["title"], eyebrow="the problem")
    y = 150.0
    for color_name, text in spec["bullets"]:
        # accent tick bar
        reqs += rect(
            sid, _id("tick"), MARGIN, y + 6, 5, 26, SPEC[color_name], corner_radius=2
        )
        bid = _id("bul")
        reqs.append(textbox(sid, bid, MARGIN + 22, y, CONTENT_W - 22, 62))
        reqs.append({"insertText": {"objectId": bid, "text": text}})
        reqs.append(style_text(bid, 16.5, FG))
        reqs.append(para_style(bid, 0, line_spacing=118))
        y += 92
    reqs += spectrum_strip(sid)
    return reqs


def build_cards(spec):
    reqs, sid = create_slide()
    reqs.append(page_bg(sid))
    slide_header(reqs, sid, spec["title"], eyebrow="the build")
    grid = spec.get("grid", len(spec["cards"]))
    gap = 16.0
    top = 128.0
    if grid == 4:
        cw = (CONTENT_W - 3 * gap) / 4
        ch = SLIDE_H - top - 40
        positions = [(MARGIN + i * (cw + gap), top) for i in range(4)]
    else:  # 2x2
        cw = (CONTENT_W - gap) / 2
        ch = (SLIDE_H - top - 36) / 2 - gap / 2
        positions = [
            (MARGIN, top),
            (MARGIN + cw + gap, top),
            (MARGIN, top + ch + gap),
            (MARGIN + cw + gap, top + ch + gap),
        ]
    for (num, color_name, head, body), (x, y) in zip(
        spec["cards"], positions, strict=True
    ):
        reqs += rect(
            sid,
            _id("card"),
            x,
            y,
            cw,
            ch,
            CARD,
            corner_radius=10,
            line_color=CARD_EDGE,
            line_w=1,
        )
        # accent top edge
        reqs += rect(
            sid, _id("edge"), x + 18, y + 16, 40, 4, SPEC[color_name], corner_radius=2
        )
        nid = _id("num")
        reqs.append(textbox(sid, nid, x + cw - 52, y + 8, 44, 40))
        reqs.append({"insertText": {"objectId": nid, "text": num}})
        reqs.append(style_text(nid, 22, SPEC[color_name], bold=True, font=FONT_M))
        reqs.append(para_style(nid, 0, alignment="END"))
        hid = _id("head")
        reqs.append(textbox(sid, hid, x + 18, y + 34, cw - 36, 34))
        reqs.append({"insertText": {"objectId": hid, "text": head}})
        reqs.append(style_text(hid, 17, FG, bold=True, font=FONT_D))
        bid = _id("body")
        reqs.append(textbox(sid, bid, x + 18, y + 72, cw - 36, ch - 88))
        reqs.append({"insertText": {"objectId": bid, "text": body}})
        reqs.append(style_text(bid, 11.5, MUT))
        reqs.append(para_style(bid, 0, line_spacing=116))
    reqs += spectrum_strip(sid)
    return reqs


def build_stats(spec):
    reqs, sid = create_slide()
    reqs.append(page_bg(sid))
    slide_header(reqs, sid, spec["title"], eyebrow="results")
    gap = 20.0
    pw = (CONTENT_W - gap) / 2
    top = 126.0
    ph = SLIDE_H - top - 36
    for i, panel in enumerate(spec["panels"]):
        x = MARGIN + i * (pw + gap)
        accent = SPEC[panel["accent"]]
        reqs += rect(
            sid,
            _id("panel"),
            x,
            top,
            pw,
            ph,
            CARD,
            corner_radius=12,
            line_color=CARD_EDGE,
            line_w=1,
        )
        reqs += rect(sid, _id("led"), x, top, 6, ph, accent)
        # name
        nid = _id("name")
        reqs.append(textbox(sid, nid, x + 26, top + 18, pw - 46, 40))
        reqs.append({"insertText": {"objectId": nid, "text": panel["name"]}})
        reqs.append(style_text(nid, 15, FG, bold=True, font=FONT_D))
        reqs.append(para_style(nid, 0, line_spacing=105))
        sid_src = _id("src")
        reqs.append(textbox(sid, sid_src, x + 26, top + 58, pw - 46, 18))
        reqs.append({"insertText": {"objectId": sid_src, "text": panel["src"]}})
        reqs.append(style_text(sid_src, 10, MUT, font=FONT_M))
        # big number
        bid = _id("big")
        reqs.append(textbox(sid, bid, x + 26, top + 86, 170, 72))
        reqs.append({"insertText": {"objectId": bid, "text": panel["big"]}})
        reqs.append(style_text(bid, 46, accent, bold=True, font=FONT_D))
        blid = _id("biglabel")
        reqs.append(textbox(sid, blid, x + 26, top + 158, 170, 34))
        reqs.append({"insertText": {"objectId": blid, "text": panel["big_label"]}})
        reqs.append(style_text(blid, 11, MUT, bold=True))
        # metrics (right column)
        my = top + 86
        for mval, mlab in panel["metrics"]:
            mid = _id("m")
            reqs.append(textbox(sid, mid, x + 200, my, pw - 230, 40))
            reqs.append({"insertText": {"objectId": mid, "text": f"{mval}\n{mlab}"}})
            reqs.append(
                style_text(mid, 13.5, FG, bold=True, font=FONT_M, end=len(mval))
            )
            reqs.append(style_text(mid, 10, MUT, start=len(mval) + 1))
            my += 44
        # judge line
        jid = _id("judge")
        reqs.append(textbox(sid, jid, x + 26, top + ph - 34, pw - 46, 22))
        reqs.append({"insertText": {"objectId": jid, "text": panel["judge"]}})
        reqs.append(style_text(jid, 10.5, SPEC["teal"], bold=True, font=FONT_M))
    reqs += spectrum_strip(sid)
    return reqs


def build_figure(spec):
    reqs, sid = create_slide()
    reqs.append(page_bg(sid))
    slide_header(reqs, sid, spec["title"], eyebrow="evidence")
    iw, ih = spec["image_dims"]
    max_w, max_h = 560.0, 236.0
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    fx = MARGIN
    fy = 118.0
    pad = 14.0
    # white plate behind the (white-bg) matplotlib figure
    reqs += rect(
        sid,
        _id("plate"),
        fx - pad,
        fy - pad,
        w + 2 * pad,
        h + 2 * pad,
        WHITE,
        corner_radius=8,
    )
    reqs.append(
        {
            "createImage": {
                "url": spec["image"],
                "elementProperties": {
                    "pageObjectId": sid,
                    "size": _size(w, h),
                    "transform": _transform(fx, fy),
                },
            }
        }
    )
    # right rail: claim chips + verdict caption
    rail_x = MARGIN + w + 40.0
    rail_w = SLIDE_W - MARGIN - rail_x
    cy = fy
    for label, color_name, note in spec["chips"]:
        reqs += rect(
            sid,
            _id("chip"),
            rail_x,
            cy,
            rail_w,
            62,
            CARD,
            corner_radius=8,
            line_color=CARD_EDGE,
            line_w=1,
        )
        reqs += rect(
            sid,
            _id("dot"),
            rail_x + 14,
            cy + 12,
            8,
            8,
            SPEC[color_name],
            corner_radius=4,
        )
        lid = _id("chiplabel")
        reqs.append(textbox(sid, lid, rail_x + 32, cy + 7, rail_w - 44, 20))
        reqs.append({"insertText": {"objectId": lid, "text": label}})
        reqs.append(style_text(lid, 12.5, FG, bold=True, font=FONT_M))
        nid = _id("chipnote")
        reqs.append(textbox(sid, nid, rail_x + 32, cy + 27, rail_w - 44, 30))
        reqs.append({"insertText": {"objectId": nid, "text": note}})
        reqs.append(style_text(nid, 10, MUT))
        reqs.append(para_style(nid, 0, line_spacing=110))
        cy += 74
    # caption under figure
    cid = _id("cap")
    reqs.append(textbox(sid, cid, MARGIN, fy + h + 26, SLIDE_W - 2 * MARGIN, 60))
    reqs.append({"insertText": {"objectId": cid, "text": spec["caption"]}})
    reqs.append(style_text(cid, 11.5, DIM))
    reqs.append(para_style(cid, 0, line_spacing=118))
    reqs += spectrum_strip(sid)
    return reqs


def build_timeline(spec):
    reqs, sid = create_slide()
    reqs.append(page_bg(sid))
    full_bleed(reqs, sid, spec["art"])
    scrim(reqs, sid)
    slide_header(reqs, sid, spec["title"], eyebrow="the turning point")
    y = 150.0
    # vertical rail
    reqs += rect(sid, _id("rail"), MARGIN + 7, y + 6, 2, 236, CARD_EDGE)
    for head, body, color_name in spec["steps"]:
        accent = SPEC[color_name]
        reqs += rect(sid, _id("node"), MARGIN, y + 4, 16, 16, accent, corner_radius=8)
        hid = _id("thead")
        reqs.append(textbox(sid, hid, MARGIN + 34, y, 320, 22))
        reqs.append({"insertText": {"objectId": hid, "text": head.upper()}})
        reqs.append(style_text(hid, 11, accent, bold=True, font=FONT_M))
        bid = _id("tbody")
        reqs.append(textbox(sid, bid, MARGIN + 34, y + 20, 560, 40))
        reqs.append({"insertText": {"objectId": bid, "text": body}})
        reqs.append(style_text(bid, 13, FG))
        reqs.append(para_style(bid, 0, line_spacing=112))
        y += 62
    pid = _id("punch")
    reqs.append(textbox(sid, pid, MARGIN, y + 14, CONTENT_W, 40))
    reqs.append(
        {"insertText": {"objectId": pid, "text": "“" + spec["punchline"] + "”"}}
    )
    reqs.append(style_text(pid, 16, SPEC["gold"], bold=True, font=FONT_D))
    reqs += spectrum_strip(sid, SLIDE_H - STRIP_H)
    return reqs


def build_links(spec):
    reqs, sid = create_slide()
    reqs.append(page_bg(sid))
    full_bleed(reqs, sid, spec["art"])
    scrim(reqs, sid)
    slide_header(reqs, sid, spec["title"], eyebrow="artifacts")
    y = 138.0
    for label, url in spec["links"]:
        reqs += rect(
            sid,
            _id("row"),
            MARGIN,
            y,
            CONTENT_W,
            56,
            CARD,
            corner_radius=8,
            line_color=CARD_EDGE,
            line_w=1,
        )
        lid = _id("label")
        reqs.append(textbox(sid, lid, MARGIN + 20, y + 8, 396, 40))
        reqs.append({"insertText": {"objectId": lid, "text": label}})
        reqs.append(style_text(lid, 13, FG, bold=True))
        uid = _id("url")
        reqs.append(textbox(sid, uid, MARGIN + 424, y + 17, CONTENT_W - 430, 22))
        short = url.split("//", 1)[-1]
        reqs.append({"insertText": {"objectId": uid, "text": short}})
        reqs.append(style_text(uid, 9.5, SPEC["teal"], font=FONT_M, end=len(short)))
        reqs.append(
            {
                "updateTextStyle": {
                    "objectId": uid,
                    "textRange": {"type": "ALL"},
                    "style": {"link": {"url": url}},
                    "fields": "link",
                }
            }
        )
        reqs.append(para_style(uid, 0, alignment="END"))
        y += 66
    reqs += spectrum_strip(sid, SLIDE_H - STRIP_H)
    return reqs


def build_closer(spec):
    reqs, sid = create_slide()
    reqs.append(page_bg(sid))
    slide_header(reqs, sid, spec["title"], eyebrow="roadmap")
    y = 130.0
    for letter, color_name, body in spec["lines"]:
        reqs += rect(
            sid, _id("badge"), MARGIN, y, 34, 34, SPEC[color_name], corner_radius=17
        )
        bid = _id("letter")
        reqs.append(textbox(sid, bid, MARGIN, y + 5, 34, 26))
        reqs.append({"insertText": {"objectId": bid, "text": letter}})
        reqs.append(style_text(bid, 16, INK, bold=True, font=FONT_D))
        reqs.append(para_style(bid, 0, alignment="CENTER"))
        tid = _id("line")
        reqs.append(textbox(sid, tid, MARGIN + 52, y + 4, CONTENT_W - 52, 34))
        reqs.append({"insertText": {"objectId": tid, "text": body}})
        reqs.append(style_text(tid, 13.5, FG))
        reqs.append(para_style(tid, 0, line_spacing=110))
        y += 54
    # thanks band
    reqs += rect(
        sid,
        _id("band"),
        MARGIN,
        y + 26,
        CONTENT_W,
        64,
        CARD,
        corner_radius=10,
        line_color=CARD_EDGE,
        line_w=1,
    )
    tid = _id("thanks")
    reqs.append(textbox(sid, tid, MARGIN, y + 44, CONTENT_W, 30))
    reqs.append({"insertText": {"objectId": tid, "text": spec["thanks"]}})
    reqs.append(style_text(tid, 15, FG, bold=True, font=FONT_D))
    reqs.append(para_style(tid, 0, alignment="CENTER"))
    reqs += spectrum_strip(sid)
    return reqs


BUILDERS = {
    "hero": build_hero,
    "bullets": build_bullets,
    "cards": build_cards,
    "stats": build_stats,
    "figure": build_figure,
    "timeline": build_timeline,
    "links": build_links,
    "closer": build_closer,
}


def main() -> int:
    creds = get_credentials()
    slides_api = build("slides", "v1", credentials=creds)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", help="rebuild an existing presentation")
    args = ap.parse_args()

    if args.pid:
        pid = args.pid
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
            print(f"  slide {i + 1}/{len(SLIDES)} FAILED — {str(exc)[:300]}")

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
