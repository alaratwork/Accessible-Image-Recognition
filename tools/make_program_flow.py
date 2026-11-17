# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# Does: draw program_flow.mermaid as a picture - the same plain-language flow,
#       as an SVG for Figma or slides and a PNG for anything that cannot read
#       SVG
# Does not: parse the mermaid file. the boxes below are written out again here
#       because there is no mermaid renderer on this machine, so the two files
#       have to be kept in step by hand if either changes. the PNG is drawn
#       directly rather than rasterised from the SVG, because QuickLook pads
#       and crops non-square SVGs unpredictably and clipped this one twice
# Reasoning: the mermaid file is the readable source and belongs in the repo
#       next to the other two diagrams, but nothing installed here can render
#       it, and a diagram nobody can see is not much use in a presentation.
#
# run from the project root:  python3 tools/make_program_flow.py

# imports start
import os
# imports end


# useful functions start

INK, MUTED, WHITE = '#1f2933', '#5b6b76', '#ffffff'
TEAL, TEAL_BG = '#0e7c86', '#dbeef0'
GREEN, GREEN_BG = '#3f7d5c', '#dff0e4'
CORAL, CORAL_BG = '#c9483f', '#fbe6e3'
GREY, GREY_BG = '#5b6b76', '#f2f5f6'

W, H = 1180, 1500
BOX_W, BOX_H, DIA_W, DIA_H = 330, 78, 250, 96

# every box: (kind, centre x, centre y, lines of text)
# kind picks the colours and the shape, matching program_flow.mermaid
BOXES = [
    ('user', 300, 90,  ['you photograph one or more pantry items', 'on a plain, dark surface']),
    ('work', 300, 210, ['the program looks for the items', 'in the photo']),
    ('work', 300, 330, ['it separates each item into its', 'own small picture']),
    ('dia',  300, 460, ['did it find', 'anything?']),
    ('careful', 790, 460, ['it says so out loud, and asks', 'for a plainer background']),
    ('work', 300, 600, ['for each item it found: compare', 'against the nine items it learned']),
    ('dia',  300, 730, ['is it sure', 'enough?']),
    ('good', 205, 870, ['it names the item', 'out loud']),
    ('careful', 620, 870, ['it says it is not sure, gives its best', 'guess, and asks you to double check']),
    ('user', 300, 1000, ['every answer ends the same way: this is a', 'prototype, not a verified identification']),
]

# the learning stage, drawn as a separate band because it happens once, before
# any of the above
LEARNED = [
    ('user', 175, 1300, ['photos of the nine', 'items, taken by hand']),
    ('work', 500, 1300, ['each photo trimmed', 'down to just the item']),
    ('work', 850, 1300, ['it studies those and remembers', 'what each item looks like']),
]

STYLE = {
    'user':    (TEAL_BG, TEAL, INK),
    'work':    (GREY_BG, GREY, INK),
    'good':    (GREEN_BG, GREEN, INK),
    'careful': (CORAL_BG, CORAL, INK),
    'dia':     (GREY_BG, GREY, INK),
}


# one rounded box or diamond with centred text
def box(kind, cx, cy, lines, width=None):
    fill, stroke, text = STYLE[kind]
    out = []
    if kind == 'dia':
        w, h = DIA_W, DIA_H
        pts = '{},{} {},{} {},{} {},{}'.format(cx, cy - h / 2, cx + w / 2, cy, cx, cy + h / 2, cx - w / 2, cy)
        out.append('<polygon points="{}" fill="{}" stroke="{}" stroke-width="2"/>'.format(pts, fill, stroke))
    else:
        w, h = (width or BOX_W), BOX_H
        out.append('<rect x="{}" y="{}" width="{}" height="{}" rx="10" fill="{}" stroke="{}" '
                   'stroke-width="2"/>'.format(cx - w / 2, cy - h / 2, w, h, fill, stroke))
    start = cy - (len(lines) - 1) * 10
    for i, line in enumerate(lines):
        out.append('<text x="{}" y="{}" font-size="15" fill="{}" text-anchor="middle">{}</text>'.format(
            cx, start + i * 20 + 5, text, line))
    return out


# straight connector with an arrow head, optionally labelled
def arrow(x1, y1, x2, y2, label=None):
    out = ['<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2" '
           'marker-end="url(#tip)"/>'.format(x1, y1, x2, y2, '#8fa0aa')]
    if label:
        out.append('<text x="{}" y="{}" font-size="13" fill="{}" text-anchor="middle">{}</text>'.format(
            (x1 + x2) / 2 + 16, (y1 + y2) / 2 - 6, MUTED, label))
    return out
# useful functions end


# main function start

def main():
    o = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}" '
         'font-family="Arial, Helvetica, sans-serif">'.format(W, H, W, H),
         '<rect width="{}" height="{}" fill="{}"/>'.format(W, H, WHITE),
         '<defs><marker id="tip" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
         '<path d="M0,0 L0,6 L7,3 z" fill="#8fa0aa"/></marker></defs>']

    o.append('<text x="40" y="46" font-size="16" font-weight="bold" letter-spacing="2" fill="{}">'
             'WHAT HAPPENS WHEN IT RUNS</text>'.format(TEAL))

    for kind, cx, cy, lines in BOXES:
        o += box(kind, cx, cy, lines, width=400 if cy == 1000 else None)

    o += arrow(300, 129, 300, 171)
    o += arrow(300, 249, 300, 291)
    o += arrow(300, 369, 300, 412)
    o += arrow(425, 460, 625, 460, 'no')
    o += arrow(300, 508, 300, 561, 'yes')
    o += arrow(300, 639, 300, 682)
    o += arrow(262, 768, 215, 831, 'yes')
    o += arrow(338, 768, 560, 831, 'no')
    o += arrow(205, 909, 260, 961)
    o += arrow(620, 909, 400, 961)

    # the learning band
    o.append('<rect x="40" y="1180" width="{}" height="200" rx="12" fill="none" stroke="#c7d0d6" '
             'stroke-width="1.5" stroke-dasharray="7 6"/>'.format(W - 80))
    o.append('<text x="62" y="1212" font-size="14" font-weight="bold" fill="{}">'
             'what it learned from, once, before any of this</text>'.format(MUTED))
    for kind, cx, cy, lines in LEARNED:
        o += box(kind, cx, cy, lines, width=290)
    o += arrow(320, 1300, 352, 1300)
    o += arrow(648, 1300, 702, 1300)
    o.append('<path d="M 850 1261 C 850 1120, 700 700, 470 600" fill="none" stroke="#8fa0aa" '
             'stroke-width="2" stroke-dasharray="6 5" marker-end="url(#tip)"/>')
    o.append('<text x="880" y="1120" font-size="13" fill="{}">this is the memory</text>'.format(MUTED))
    o.append('<text x="880" y="1140" font-size="13" fill="{}">the comparison uses</text>'.format(MUTED))

    o.append('</svg>')
    open('diagrams/program_flow.svg', 'w').write('\n'.join(o))
    print('wrote diagrams/program_flow.svg ({} x {})'.format(W, H))

    # draw the PNG directly rather than rasterising the SVG. QuickLook pads
    # and crops non-square SVGs unpredictably, and it clipped the bottom band
    # twice; drawing from the same coordinate table cannot disagree with itself.
    from PIL import Image, ImageDraw, ImageFont
    scale = 2  # draw at 2x so the text stays crisp when placed in a slide

    def font_at(size):
        for path in ['/System/Library/Fonts/Supplemental/Arial.ttf',
                     '/System/Library/Fonts/Helvetica.ttc']:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    img = Image.new('RGB', (W * scale, 1420 * scale), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f_box, f_small, f_title = font_at(15 * scale), font_at(13 * scale), font_at(16 * scale)

    def px(v):
        return v * scale

    def draw_text_centred(cx, cy, lines, colour, font):
        start_y = cy - (len(lines) - 1) * 10
        for i, line in enumerate(lines):
            d.text((px(cx), px(start_y + i * 20)), line, font=font, fill=colour, anchor='mm')

    def draw_box(kind, cx, cy, lines, width=None):
        fill, stroke, text = STYLE[kind]
        if kind == 'dia':
            w, h = DIA_W, DIA_H
            d.polygon([(px(cx), px(cy - h / 2)), (px(cx + w / 2), px(cy)),
                       (px(cx), px(cy + h / 2)), (px(cx - w / 2), px(cy))],
                      fill=fill, outline=stroke, width=2 * scale)
        else:
            w, h = (width or BOX_W), BOX_H
            d.rounded_rectangle([px(cx - w / 2), px(cy - h / 2), px(cx + w / 2), px(cy + h / 2)],
                                radius=10 * scale, fill=fill, outline=stroke, width=2 * scale)
        draw_text_centred(cx, cy, lines, text, f_box)

    def draw_arrow(x1, y1, x2, y2, label=None):
        d.line([px(x1), px(y1), px(x2), px(y2)], fill='#8fa0aa', width=2 * scale)
        # arrow head, pointing along the line
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        size = 9
        for sign in (1, -1):
            bx = x2 - size * math.cos(ang - sign * 0.45)
            by = y2 - size * math.sin(ang - sign * 0.45)
            d.line([px(x2), px(y2), px(bx), px(by)], fill='#8fa0aa', width=2 * scale)
        if label:
            d.text((px((x1 + x2) / 2 + 16), px((y1 + y2) / 2 - 6)), label,
                   font=f_small, fill=MUTED, anchor='mm')

    d.text((px(40), px(40)), 'WHAT HAPPENS WHEN IT RUNS', font=f_title, fill=TEAL, anchor='lm')
    for kind, cx, cy, lines in BOXES:
        draw_box(kind, cx, cy, lines, width=400 if cy == 1000 else None)
    for a in [(300, 129, 300, 171, None), (300, 249, 300, 291, None), (300, 369, 300, 412, None),
              (425, 460, 625, 460, 'no'), (300, 508, 300, 561, 'yes'), (300, 639, 300, 682, None),
              (262, 768, 215, 831, 'yes'), (338, 768, 560, 831, 'no'),
              (205, 909, 260, 961, None), (620, 909, 400, 961, None)]:
        draw_arrow(*a)

    d.rounded_rectangle([px(40), px(1180), px(W - 40), px(1380)], radius=12 * scale,
                        outline='#c7d0d6', width=2 * scale)
    d.text((px(62), px(1208)), 'what it learned from, once, before any of this',
           font=f_small, fill=MUTED, anchor='lm')
    for kind, cx, cy, lines in LEARNED:
        draw_box(kind, cx, cy, lines, width=290)
    draw_arrow(320, 1300, 352, 1300)
    draw_arrow(648, 1300, 702, 1300)
    draw_arrow(850, 1261, 500, 645)
    d.text((px(880), px(1120)), 'this is the memory', font=f_small, fill=MUTED, anchor='lm')
    d.text((px(880), px(1140)), 'the comparison uses', font=f_small, fill=MUTED, anchor='lm')

    img.save('diagrams/program_flow.png')
    print('wrote diagrams/program_flow.png', img.size)
    return

if __name__ == "__main__":
    main()
# main function end
