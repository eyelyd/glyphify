use anyhow::Result;
use clap::Parser;

mod eng;

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    input: String,

    #[arg(short, long, default_value = "out.html")]
    output: String,

    #[arg(short, long, default_value_t = 160)]
    width: u32,

    #[arg(long, default_value_t = 0.55)]
    ratio: f32,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    dither: bool,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    color: bool,

    #[arg(long, default_value = "braille")]
    mode: String,

    #[arg(long)]
    ttf: Option<String>,

    #[arg(long, default_value_t = 16.0)]
    px: f32,

    #[arg(long, default_value = "ramp")]
    set: String,

    #[arg(long, default_value_t = 1.0)]
    zoom: f32,

    #[arg(long, default_value_t = 0.35)]
    edge: f32,

    #[arg(long, default_value_t = 0.25)]
    dir: f32,

    #[arg(
        long,
        default_value = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace"
    )]
    font: String,
}

fn main() {
    if let Err(err) = _run() {
        eprintln!("{err:#}");
        std::process::exit(1);
    }
}

fn _run() -> Result<()> {
    let args = Args::parse();

    let html = eng::_make(
        &args.input,
        args.width,
        args.ratio,
        args.dither,
        args.color,
        &args.font,
        &args.mode,
        args.ttf.as_deref(),
        args.px,
        &args.set,
        args.zoom,
        args.edge,
        args.dir,
    )?;
    std::fs::write(&args.output, html)?;

    Ok(())
}
