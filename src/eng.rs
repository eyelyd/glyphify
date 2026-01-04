use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use fontdue::{Font, FontSettings};
use image::{GenericImageView, ImageBuffer, Rgba};
use std::path::Path;

pub fn _make(
    path: &str,
    width: u32,
    ratio: f32,
    dither: bool,
    color: bool,
    font: &str,
    mode: &str,
    ttf: Option<&str>,
    px: f32,
    set: &str,
    zoom: f32,
    edge: f32,
    dir: f32,
) -> Result<String> {
    if mode == "glyph" {
        return _gly(
            path, width, ratio, dither, color, font, ttf, px, set, zoom, edge, dir,
        );
    }

    _brl(path, width, ratio, dither, color, font, zoom)
}

fn _brl(
    path: &str,
    width: u32,
    ratio: f32,
    dither: bool,
    color: bool,
    font: &str,
    zoom: f32,
) -> Result<String> {
    let img = image::open(path)?;
    let (origw, origh) = img.dimensions();

    if origw == 0 || origh == 0 {
        return Err(anyhow!("bad image dimensions"));
    }

    let mut w = width.saturating_mul(2).max(2);
    let mut h = ((origh as f32 / origw as f32) * w as f32 * ratio).round() as u32;
    h = h.max(4);

    /* align to braille cell grid */
    w = (w + 1) / 2 * 2;
    h = (h + 3) / 4 * 4;

    let img = img.to_rgba8();
    let img = image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3);

    let mut lum = vec![0.0f32; (w as usize) * (h as usize)];

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y);
            let (r, g, b, a) = (px[0] as f32, px[1] as f32, px[2] as f32, px[3] as f32);

            let af = a / 255.0;
            let r = r * af + 255.0 * (1.0 - af);
            let g = g * af + 255.0 * (1.0 - af);
            let b = b * af + 255.0 * (1.0 - af);

            let v = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
            let i = (y as usize) * (w as usize) + (x as usize);
            lum[i] = v.clamp(0.0, 1.0);
        }
    }

    if dither {
        _dither(&mut lum, w as usize, h as usize);
    }

    let cw = (w / 2) as usize;
    let ch = (h / 4) as usize;

    let mut out = String::new();

    out.push_str("<!doctype html>\n");
    out.push_str("<html>\n");
    out.push_str("<head>\n");
    out.push_str("<meta charset=\"utf-8\">\n");
    out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    out.push_str("<title>glyphify</title>\n");
    out.push_str("<style>\n");
    out.push_str("body{margin:0;background:#000;}\n");
    out.push_str("pre{margin:0;padding:12px;color:#fff;display:inline-block;font-family:");
    out.push_str(font);
    out.push_str(";font-size:8px;line-height:8px;white-space:pre;");
    if zoom != 1.0 {
        out.push_str("transform:scale(");
        out.push_str(&format!("{}", zoom));
        out.push_str(");transform-origin:0 0;");
    }
    out.push_str("}\n");
    out.push_str("</style>\n");
    out.push_str("</head>\n");
    out.push_str("<body>\n");
    out.push_str("<pre>\n");

    for y in 0..ch {
        let mut last: Option<[u8; 6]> = None;
        let mut open = false;

        for x in 0..cw {
            let bits = _cell(&lum, w as usize, x, y);

            let (key, span) = if color {
                let key = _col(&img, &lum, w as usize, x, y);
                (Some(key), Some(_span(key)))
            } else {
                (None, None)
            };

            if let Some(key) = key {
                if last.map(|v| v != key).unwrap_or(true) {
                    if open {
                        out.push_str("</span>");
                    }
                    out.push_str(&span.unwrap());
                    open = true;
                    last = Some(key);
                }
            } else if open {
                out.push_str("</span>");
                open = false;
                last = None;
            }

            let chr = _braille(bits)?;
            out.push(chr);
        }

        if open {
            out.push_str("</span>");
        }

        out.push('\n');
    }

    out.push_str("</pre>\n");
    out.push_str("</body>\n");
    out.push_str("</html>\n");

    Ok(out)
}

fn _dither(lum: &mut [f32], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let old = lum[i];
            let new = if old >= 0.5 { 1.0 } else { 0.0 };
            let err = old - new;
            lum[i] = new;

            /* floyd steinberg dithering */
            if x + 1 < w {
                lum[i + 1] = (lum[i + 1] + err * (7.0 / 16.0)).clamp(0.0, 1.0);
            }
            if y + 1 < h {
                if x > 0 {
                    lum[i + w - 1] = (lum[i + w - 1] + err * (3.0 / 16.0)).clamp(0.0, 1.0);
                }
                lum[i + w] = (lum[i + w] + err * (5.0 / 16.0)).clamp(0.0, 1.0);
                if x + 1 < w {
                    lum[i + w + 1] = (lum[i + w + 1] + err * (1.0 / 16.0)).clamp(0.0, 1.0);
                }
            }
        }
    }
}

fn _cell(lum: &[f32], w: usize, x: usize, y: usize) -> u8 {
    let mut bits: u8 = 0;

    for dy in 0..4usize {
        for dx in 0..2usize {
            let px = x * 2 + dx;
            let py = y * 4 + dy;
            let i = py * w + px;

            let on = lum[i] < 0.5;

            if on {
                bits |= _bit(dx, dy);
            }
        }
    }

    bits
}

fn _bit(x: usize, y: usize) -> u8 {
    match (x, y) {
        (0, 0) => 1 << 0,
        (0, 1) => 1 << 1,
        (0, 2) => 1 << 2,
        (1, 0) => 1 << 3,
        (1, 1) => 1 << 4,
        (1, 2) => 1 << 5,
        (0, 3) => 1 << 6,
        (1, 3) => 1 << 7,
        _ => 0,
    }
}

fn _braille(bits: u8) -> Result<char> {
    let code = 0x2800u32 + bits as u32;
    std::char::from_u32(code).ok_or_else(|| anyhow!("bad braille code"))
}

fn _col(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    lum: &[f32],
    w: usize,
    x: usize,
    y: usize,
) -> [u8; 6] {
    let mut onr: u32 = 0;
    let mut ong: u32 = 0;
    let mut onb: u32 = 0;
    let mut onc: u32 = 0;

    let mut offr: u32 = 0;
    let mut offg: u32 = 0;
    let mut offb: u32 = 0;
    let mut offc: u32 = 0;

    for dy in 0..4usize {
        for dx in 0..2usize {
            let px = (x * 2 + dx) as u32;
            let py = (y * 4 + dy) as u32;
            let i = (py as usize) * w + (px as usize);

            let on = lum[i] < 0.5;

            let px = img.get_pixel(px, py);
            let (r, g, b) = (px[0] as u32, px[1] as u32, px[2] as u32);

            if on {
                onr += r;
                ong += g;
                onb += b;
                onc += 1;
            } else {
                offr += r;
                offg += g;
                offb += b;
                offc += 1;
            }
        }
    }

    let (fgr, fgg, fgb) = if onc > 0 {
        ((onr / onc) as u8, (ong / onc) as u8, (onb / onc) as u8)
    } else {
        (0u8, 0u8, 0u8)
    };

    let (bgr, bgg, bgb) = if offc > 0 {
        (
            (offr / offc) as u8,
            (offg / offc) as u8,
            (offb / offc) as u8,
        )
    } else {
        (fgr, fgg, fgb)
    };

    let (fgr, fgg, fgb) = if onc == 0 {
        (bgr, bgg, bgb)
    } else {
        (fgr, fgg, fgb)
    };

    [fgr, fgg, fgb, bgr, bgg, bgb]
}

fn _span(key: [u8; 6]) -> String {
    let (fgr, fgg, fgb, bgr, bgg, bgb) = (key[0], key[1], key[2], key[3], key[4], key[5]);

    format!(
        "<span style=\"color:rgb({},{},{});background-color:rgb({},{},{});\">",
        fgr, fgg, fgb, bgr, bgg, bgb
    )
}

struct Glyph {
    chr: char,
    mean: f32,
    mask: Vec<f32>,
    dx: Vec<f32>,
    dy: Vec<f32>,
}

fn _gly(
    path: &str,
    width: u32,
    ratio: f32,
    dither: bool,
    color: bool,
    font: &str,
    ttf: Option<&str>,
    px: f32,
    set: &str,
    zoom: f32,
    edge: f32,
    dir: f32,
) -> Result<String> {
    let img = image::open(path)?;
    let (iw, ih) = img.dimensions();

    if iw == 0 || ih == 0 {
        return Err(anyhow!("bad image dimensions"));
    }

    let mut ttf = ttf;
    if ttf.is_none() {
        let sys = "C:\\Windows\\Fonts\\consola.ttf";
        if Path::new(sys).exists() {
            ttf = Some(sys);
        }
    }

    let ttf = ttf.ok_or_else(|| anyhow!("ttf is required for glyph mode"))?;
    let data = std::fs::read(ttf)?;
    let face =
        Font::from_bytes(data.clone(), FontSettings::default()).map_err(|err| anyhow!(err))?;

    let line = face
        .horizontal_line_metrics(px)
        .ok_or_else(|| anyhow!("missing font line metrics"))?;
    let base = line.ascent.ceil() as i32;

    let mut refc = 'M';
    if !face.has_glyph(refc) {
        refc = '0';
    }
    if !face.has_glyph(refc) {
        refc = '#';
    }
    let met = face.metrics(refc, px);

    let cw = (met.advance_width.ceil() as i32).max(1) as u32;
    let ch = (line.new_line_size.ceil() as i32).max(1) as u32;

    let cols = width.max(1);
    let mut rw = cols.saturating_mul(cw).max(1);
    let mut rh = ((ih as f32 / iw as f32) * rw as f32 * ratio).round() as u32;
    rh = rh.max(ch);

    rh = (rh / ch).max(1) * ch;
    rw = (rw / cw).max(1) * cw;

    let img = img.to_rgba8();
    let img = image::imageops::resize(&img, rw, rh, image::imageops::FilterType::Lanczos3);

    let w = rw as usize;
    let h = rh as usize;
    let cw = cw as usize;
    let ch = ch as usize;

    let mut dark = vec![0.0f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x as u32, y as u32);
            let (r, g, b, a) = (px[0] as f32, px[1] as f32, px[2] as f32, px[3] as f32);

            let af = a / 255.0;
            let r = r * af + 255.0 * (1.0 - af);
            let g = g * af + 255.0 * (1.0 - af);
            let b = b * af + 255.0 * (1.0 - af);

            let v = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
            let i = y * w + x;
            let v = v.clamp(0.0, 1.0);
            dark[i] = 1.0 - v;
        }
    }

    let useg = edge > 0.0 || dir > 0.0;
    let (dx, dy) = if useg {
        _vec(&dark, w, h)
    } else {
        (Vec::new(), Vec::new())
    };

    let set = _set(set);
    let mut glyphs: Vec<Glyph> = Vec::new();
    for chr in set {
        if !face.has_glyph(chr) {
            continue;
        }

        let (met, bit) = face.rasterize(chr, px);
        let mut mask = vec![0.0f32; cw * ch];

        if met.width > 0 && met.height > 0 {
            let x0 = met.xmin;
            let y0 = base - (met.ymin + met.height as i32);

            for by in 0..met.height {
                for bx in 0..met.width {
                    let x = x0 + bx as i32;
                    let y = y0 + by as i32;

                    if x < 0 || y < 0 {
                        continue;
                    }
                    let x = x as usize;
                    let y = y as usize;
                    if x >= cw || y >= ch {
                        continue;
                    }

                    let i = y * cw + x;
                    let j = by * met.width + bx;
                    mask[i] = (bit[j] as f32) / 255.0;
                }
            }
        }

        let mean = mask.iter().sum::<f32>() / (mask.len() as f32);
        let (gdx, gdy) = if useg {
            _vec(&mask, cw, ch)
        } else {
            (Vec::new(), Vec::new())
        };
        glyphs.push(Glyph {
            chr,
            mean,
            mask,
            dx: gdx,
            dy: gdy,
        });
    }

    if glyphs.is_empty() {
        return Err(anyhow!("no glyphs available"));
    }

    glyphs.sort_by(|a, b| {
        a.mean
            .partial_cmp(&b.mean)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let rows = (rh / (ch as u32)).max(1) as usize;
    let cols = (rw / (cw as u32)).max(1) as usize;

    let mut tile = vec![0.0f32; rows * cols];

    for y in 0..rows {
        for x in 0..cols {
            let mut sum = 0.0f32;
            for dy in 0..ch {
                let py = y * ch + dy;
                let base = py * w + x * cw;
                for dx in 0..cw {
                    sum += dark[base + dx];
                }
            }

            tile[y * cols + x] = sum / ((cw * ch) as f32);
        }
    }

    let name = "glyphify";
    let b64 = general_purpose::STANDARD.encode(data);

    let mut out = String::new();

    out.push_str("<!doctype html>\n");
    out.push_str("<html>\n");
    out.push_str("<head>\n");
    out.push_str("<meta charset=\"utf-8\">\n");
    out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    out.push_str("<title>glyphify</title>\n");
    out.push_str("<style>\n");
    out.push_str("@font-face{font-family:");
    out.push_str(name);
    out.push_str(";src:url(data:font/ttf;base64,");
    out.push_str(&b64);
    out.push_str(") format('truetype');font-weight:normal;font-style:normal;}\n");
    out.push_str("body{margin:0;background:#000;}\n");
    out.push_str("pre{margin:0;padding:12px;color:#fff;display:inline-block;font-variant-ligatures:none;font-family:");
    out.push_str(name);
    out.push(',');
    out.push_str(font);
    out.push_str(";font-size:");
    out.push_str(&format!("{}px", (px.round() as i32).max(1)));
    out.push_str(";line-height:");
    out.push_str(&format!("{}px", ch));
    out.push_str(";white-space:pre;");
    if zoom != 1.0 {
        out.push_str("transform:scale(");
        out.push_str(&format!("{}", zoom));
        out.push_str(");transform-origin:0 0;");
    }
    out.push_str("}\n");
    out.push_str("</style>\n");
    out.push_str("</head>\n");
    out.push_str("<body>\n");
    out.push_str("<pre>\n");

    for y in 0..rows {
        let mut last: Option<[u8; 6]> = None;
        let mut open = false;

        for x in 0..cols {
            let i = y * cols + x;
            let val = tile[i].clamp(0.0, 1.0);
            let pos = _find(&glyphs, val);

            let mut best = 0usize;
            let mut score = f32::INFINITY;

            let rad = 10usize;
            let beg = pos.saturating_sub(rad);
            let end = (pos + rad).min(glyphs.len());
            for j in beg..end {
                let cur = _score(
                    &dark,
                    &dx,
                    &dy,
                    w,
                    x * cw,
                    y * ch,
                    cw,
                    ch,
                    &glyphs[j].mask,
                    &glyphs[j].dx,
                    &glyphs[j].dy,
                    edge,
                    dir,
                    score,
                );
                if cur < score {
                    score = cur;
                    best = j;
                }
            }

            let glyph = &glyphs[best];

            if dither {
                let err = val - glyph.mean;
                if x + 1 < cols {
                    tile[i + 1] = (tile[i + 1] + err * (7.0 / 16.0)).clamp(0.0, 1.0);
                }
                if y + 1 < rows {
                    if x > 0 {
                        tile[i + cols - 1] =
                            (tile[i + cols - 1] + err * (3.0 / 16.0)).clamp(0.0, 1.0);
                    }
                    tile[i + cols] = (tile[i + cols] + err * (5.0 / 16.0)).clamp(0.0, 1.0);
                    if x + 1 < cols {
                        tile[i + cols + 1] =
                            (tile[i + cols + 1] + err * (1.0 / 16.0)).clamp(0.0, 1.0);
                    }
                }
            }

            let (key, span) = if color {
                let key = _gcol(&img, &glyph.mask, x * cw, y * ch, cw, ch);
                (Some(key), Some(_span(key)))
            } else {
                (None, None)
            };

            if let Some(key) = key {
                if last.map(|v| v != key).unwrap_or(true) {
                    if open {
                        out.push_str("</span>");
                    }
                    out.push_str(&span.unwrap());
                    open = true;
                    last = Some(key);
                }
            } else if open {
                out.push_str("</span>");
                open = false;
                last = None;
            }

            _put(&mut out, glyph.chr);
        }

        if open {
            out.push_str("</span>");
        }
        out.push('\n');
    }

    out.push_str("</pre>\n");
    out.push_str("</body>\n");
    out.push_str("</html>\n");

    Ok(out)
}

fn _set(set: &str) -> Vec<char> {
    let mut out: Vec<char> = Vec::new();

    if set == "ramp" {
        for chr in " .:-=+*#%".chars() {
            out.push(chr);
        }

        out.push('░');
        out.push('▒');
        out.push('▓');
        out.push('█');

        return out;
    }

    if set == "block" {
        out.push(' ');
        out.push('░');
        out.push('▒');
        out.push('▓');
        out.push('█');
        return out;
    }

    if set == "safe" {
        for chr in " .:-=+*#%".chars() {
            out.push(chr);
        }
        out.push('░');
        out.push('▒');
        out.push('▓');
        out.push('█');
        return out;
    }

    for chr in 32u8..=126u8 {
        out.push(chr as char);
    }

    out.push('░');
    out.push('▒');
    out.push('▓');
    out.push('█');
    out.push('▄');
    out.push('▀');
    out.push('▌');
    out.push('▐');
    out.push('■');
    out.push('●');
    out.push('○');
    out.push('◆');
    out.push('▲');
    out.push('▼');
    out.push('◀');
    out.push('▶');
    out.push('│');
    out.push('─');
    out.push('┼');
    out.push('╱');
    out.push('╲');

    out
}

fn _find(glyphs: &[Glyph], val: f32) -> usize {
    let mut lo = 0usize;
    let mut hi = glyphs.len();

    while lo < hi {
        let mid = (lo + hi) / 2;
        if glyphs[mid].mean < val {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    lo
}

fn _score(
    dark: &[f32],
    gx: &[f32],
    gy: &[f32],
    w: usize,
    x: usize,
    y: usize,
    cw: usize,
    ch: usize,
    mask: &[f32],
    hx: &[f32],
    hy: &[f32],
    edge: f32,
    dir: f32,
    lim: f32,
) -> f32 {
    let mut sum = 0.0f32;
    let mut esum = 0.0f32;
    let mut dsum = 0.0f32;
    let useg = edge > 0.0 || dir > 0.0;
    let eps = 0.02;

    for yy in 0..ch {
        let base = (y + yy) * w + x;
        let off = yy * cw;
        for xx in 0..cw {
            let d = dark[base + xx];
            let g = mask[off + xx];
            let e = d - g;
            sum += e * e;

            if useg {
                let ax = gx[base + xx];
                let ay = gy[base + xx];
                let bx = hx[off + xx];
                let by = hy[off + xx];

                if edge > 0.0 {
                    let am = ax.abs() + ay.abs();
                    let bm = bx.abs() + by.abs();
                    let ee = am - bm;
                    esum += ee * ee;
                }

                if dir > 0.0 {
                    let am = ax.abs() + ay.abs();
                    let bm = bx.abs() + by.abs();

                    if am > eps && bm > eps {
                        let anx = ax / am;
                        let any = ay / am;
                        let bnx = bx / bm;
                        let bny = by / bm;
                        let dot = (anx * bnx + any * bny).clamp(-1.0, 1.0);
                        let de = 1.0 - dot;
                        dsum += de * de;
                    }
                }
            }

            if lim.is_finite() {
                let cur = sum + esum * edge + dsum * dir;
                if cur > lim {
                    return cur;
                }
            }
        }
    }

    sum + esum * edge + dsum * dir
}

fn _vec(src: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let mut outx = vec![0.0f32; w * h];
    let mut outy = vec![0.0f32; w * h];

    for y in 0..h {
        let y0 = if y == 0 { 0 } else { y - 1 };
        let y1 = if y + 1 >= h { h - 1 } else { y + 1 };
        for x in 0..w {
            let x0 = if x == 0 { 0 } else { x - 1 };
            let x1 = if x + 1 >= w { w - 1 } else { x + 1 };

            let a = src[y * w + x0];
            let b = src[y * w + x1];
            let c = src[y0 * w + x];
            let d = src[y1 * w + x];

            outx[y * w + x] = (b - a) * 0.5;
            outy[y * w + x] = (d - c) * 0.5;
        }
    }

    (outx, outy)
}

fn _gcol(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    mask: &[f32],
    x: usize,
    y: usize,
    cw: usize,
    ch: usize,
) -> [u8; 6] {
    let mut onr: f32 = 0.0;
    let mut ong: f32 = 0.0;
    let mut onb: f32 = 0.0;
    let mut onc: f32 = 0.0;

    let mut offr: f32 = 0.0;
    let mut offg: f32 = 0.0;
    let mut offb: f32 = 0.0;
    let mut offc: f32 = 0.0;

    for dy in 0..ch {
        for dx in 0..cw {
            let g = mask[dy * cw + dx].clamp(0.0, 1.0);
            let g = g * g * g;
            let inv = 1.0 - g;

            let px = img.get_pixel((x + dx) as u32, (y + dy) as u32);
            let (r, gg, b, a) = (px[0] as f32, px[1] as f32, px[2] as f32, px[3] as f32);

            let af = a / 255.0;
            let r = r * af + 255.0 * (1.0 - af);
            let gg = gg * af + 255.0 * (1.0 - af);
            let b = b * af + 255.0 * (1.0 - af);

            onr += r * g;
            ong += gg * g;
            onb += b * g;
            onc += g;

            offr += r * inv;
            offg += gg * inv;
            offb += b * inv;
            offc += inv;
        }
    }

    let onc = onc.max(0.000001);
    let offc = offc.max(0.000001);

    let fgr = (onr / onc).round().clamp(0.0, 255.0) as u8;
    let fgg = (ong / onc).round().clamp(0.0, 255.0) as u8;
    let fgb = (onb / onc).round().clamp(0.0, 255.0) as u8;

    let bgr = (offr / offc).round().clamp(0.0, 255.0) as u8;
    let bgg = (offg / offc).round().clamp(0.0, 255.0) as u8;
    let bgb = (offb / offc).round().clamp(0.0, 255.0) as u8;

    [fgr, fgg, fgb, bgr, bgg, bgb]
}

fn _put(out: &mut String, chr: char) {
    match chr {
        '<' => out.push_str("&lt;"),
        '>' => out.push_str("&gt;"),
        '&' => out.push_str("&amp;"),
        _ => out.push(chr),
    }
}
