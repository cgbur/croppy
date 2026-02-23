use std::collections::VecDeque;
use std::io::{self, Stdout};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{fs, thread};

use anyhow::{Result, anyhow};
use clap::Parser;
use croppy::detect::BoundsNorm;
use croppy::detect_refine::{
    DetectRefineRun, RotationRefineConfig, draw_norm_rect, draw_refined_inner_backproject,
    rotate_rgb_about_center, run_detection_with_rotation_refine,
};
use croppy::discover::{is_supported_raw, list_raw_files};
use croppy::preprocess::{PreprocessConfig, prepare_image, resize_rgb_max_edge};
use croppy::raw::{decode_raw_to_rgb_with_hint, rgb_to_gray};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Gauge, List, ListItem, Paragraph};

#[derive(Parser, Debug)]
#[command(about = "Croppy TUI batch runner")]
struct Args {
    #[arg(default_value = ".")]
    input: PathBuf,

    #[arg(long, default_value_t = true)]
    recursive: bool,

    #[arg(long, default_value_t = 1000)]
    max_edge: u32,

    #[arg(long, default_value = ".")]
    out_dir: PathBuf,
}

#[derive(Clone)]
struct RunOptions {
    write_xmp: bool,
    write_preview: bool,
    preview_mode: PreviewMode,
    max_edge: u32,
    out_dir: PathBuf,
    final_crop_scale_pct: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreviewMode {
    DebugOverlay,
    FinalCrop,
    FinalCropFramed,
}

impl PreviewMode {
    fn label(self) -> &'static str {
        match self {
            Self::DebugOverlay => "Overlay",
            Self::FinalCrop => "Final Crop",
            Self::FinalCropFramed => "Crop + Frame",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::DebugOverlay => Self::FinalCrop,
            Self::FinalCrop => Self::FinalCropFramed,
            Self::FinalCropFramed => Self::DebugOverlay,
        }
    }
}

#[derive(Clone)]
struct FileResult {
    preview: Option<PathBuf>,
    xmp: Option<PathBuf>,
    error: Option<String>,
}

struct RawJob {
    raw: PathBuf,
}

struct PipelineStats {
    pending_count: AtomicUsize,
    reading_count: AtomicUsize,
    reading_peak: AtomicUsize,
    processing_count: AtomicUsize,
    processing_peak: AtomicUsize,
    buffered_count: AtomicUsize,
    buffered_peak: AtomicUsize,
    buffered_capacity: usize,
    read_workers: usize,
    process_workers: usize,
}

impl PipelineStats {
    fn new(
        pending_paths: usize,
        buffered_capacity: usize,
        read_workers: usize,
        process_workers: usize,
    ) -> Self {
        Self {
            pending_count: AtomicUsize::new(pending_paths),
            reading_count: AtomicUsize::new(0),
            reading_peak: AtomicUsize::new(0),
            processing_count: AtomicUsize::new(0),
            processing_peak: AtomicUsize::new(0),
            buffered_count: AtomicUsize::new(0),
            buffered_peak: AtomicUsize::new(0),
            buffered_capacity,
            read_workers,
            process_workers,
        }
    }

    fn mark_path_taken(&self) {
        Self::dec_counter(&self.pending_count);
    }

    fn start_reading(&self) {
        Self::inc_with_peak(&self.reading_count, &self.reading_peak);
    }

    fn finish_reading(&self) {
        Self::dec_counter(&self.reading_count);
    }

    fn start_processing(&self) {
        Self::inc_with_peak(&self.processing_count, &self.processing_peak);
    }

    fn finish_processing(&self) {
        Self::dec_counter(&self.processing_count);
    }

    fn inc_buffered(&self) {
        Self::inc_with_peak(&self.buffered_count, &self.buffered_peak);
    }

    fn dec_buffered(&self) {
        Self::dec_counter(&self.buffered_count);
    }

    fn inc_with_peak(count: &AtomicUsize, peak: &AtomicUsize) {
        let next = count.fetch_add(1, Ordering::Relaxed) + 1;
        peak.fetch_max(next, Ordering::Relaxed);
    }

    fn dec_counter(count: &AtomicUsize) {
        loop {
            let cur = count.load(Ordering::Relaxed);
            if cur == 0 {
                break;
            }
            if count
                .compare_exchange(cur, cur - 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }
}

enum WorkerMsg {
    Started(String),
    Finished(FileResult),
    Done,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Select,
    Running,
}

const PREVIEW_SUBDIR: &str = "previews";
const CANCELLED_MARKER: &str = "__croppy_cancelled__";
const FINAL_CROP_SCALE_MIN_PCT: f32 = -20.0;
const FINAL_CROP_SCALE_MAX_PCT: f32 = 20.0;
const FINAL_CROP_SCALE_FINE_STEP_PCT: f32 = 0.25;
const FRAME_PAD_FRAC: f32 = 0.06;
const FRAME_PAD_MIN_PX: u32 = 20;

struct App {
    raws: Vec<PathBuf>,
    selected: Vec<bool>,
    write_xmp: bool,
    write_preview: bool,
    preview_mode: PreviewMode,
    final_crop_scale_pct: f32,
    screen: Screen,
    file_cursor: usize,
    progress_done: usize,
    progress_total: usize,
    current_file: String,
    results: Vec<FileResult>,
    worker_rx: Option<Receiver<WorkerMsg>>,
    worker_cancel: Option<Arc<AtomicBool>>,
    cancel_requested: bool,
    run_started_at: Option<Instant>,
    select_notice: Option<String>,
    overwrite_prompt_count: Option<usize>,
    pipeline_stats: Option<Arc<PipelineStats>>,
}

#[derive(Debug, Clone, Copy)]
struct XmpCrop {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    angle_deg: f32,
}

impl App {
    fn new(raws: Vec<PathBuf>) -> Self {
        let selected = vec![false; raws.len()];
        Self {
            raws,
            selected,
            write_xmp: false,
            write_preview: false,
            preview_mode: PreviewMode::DebugOverlay,
            final_crop_scale_pct: 0.0,
            screen: Screen::Select,
            file_cursor: 0,
            progress_done: 0,
            progress_total: 0,
            current_file: String::new(),
            results: Vec::new(),
            worker_rx: None,
            worker_cancel: None,
            cancel_requested: false,
            run_started_at: None,
            select_notice: None,
            overwrite_prompt_count: None,
            pipeline_stats: None,
        }
    }

    fn selected_paths(&self) -> Vec<PathBuf> {
        self.raws
            .iter()
            .zip(self.selected.iter())
            .filter_map(|(p, s)| if *s { Some(p.clone()) } else { None })
            .collect()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let raws = discover_inputs(&args.input, args.recursive)?;
    if raws.is_empty() {
        return Err(anyhow!("no RAW files found under {}", args.input.display()));
    }

    let mut app = App::new(raws);
    let mut terminal = setup_terminal()?;
    let run_result = run_app(&mut terminal, &mut app, &args);
    teardown_terminal(&mut terminal)?;
    run_result
}

fn discover_inputs(input: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        if is_supported_raw(input) {
            return Ok(vec![input.to_path_buf()]);
        }
        return Err(anyhow!("unsupported RAW file: {}", input.display()));
    }
    list_raw_files(input, recursive)
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

fn teardown_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
    args: &Args,
) -> Result<()> {
    loop {
        let mut worker_done = false;
        if let Some(rx) = &app.worker_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    WorkerMsg::Started(s) => app.current_file = s,
                    WorkerMsg::Finished(res) => {
                        app.progress_done += 1;
                        app.results.push(res);
                    }
                    WorkerMsg::Done => worker_done = true,
                }
            }
        }
        if worker_done {
            app.worker_rx = None;
            app.worker_cancel = None;
            app.pipeline_stats = None;
            app.current_file.clear();
            app.overwrite_prompt_count = None;
            let (ok, failed, canceled) = summarize_results(&app.results);
            app.select_notice = Some(format!(
                "Run complete: {ok} ok / {failed} failed / {canceled} canceled."
            ));
            app.screen = Screen::Select;
        }

        terminal.draw(|f| {
            let size = f.area();
            match app.screen {
                Screen::Select => draw_select(f, size, app, args),
                Screen::Running => draw_running(f, size, app),
            }
        })?;

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        if let Event::Key(key) = event::read()? {
            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                return Ok(());
            }
            match app.screen {
                Screen::Select => {
                    if handle_select_key(app, args, key.code)? {
                        return Ok(());
                    }
                }
                Screen::Running => match key.code {
                    KeyCode::Char('c') => {
                        if let Some(cancel) = &app.worker_cancel {
                            cancel.store(true, Ordering::Relaxed);
                            app.cancel_requested = true;
                        }
                    }
                    KeyCode::Char('q') => return Ok(()),
                    _ => {}
                },
            }
        }
    }
}

fn draw_select(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, app: &App, args: &Args) {
    draw_screen_chrome(f, area, "Croppy", "Select Files");
    let header_subline = if let Some(count) = app.overwrite_prompt_count {
        Line::from(Span::styled(
            format!(
                "{count} selected files already have .xmp sidecars. Press y to overwrite, n to cancel."
            ),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
    } else if let Some(msg) = &app.select_notice {
        Line::from(Span::styled(
            msg.clone(),
            Style::default().fg(Color::Yellow),
        ))
    } else {
        Line::from(Span::styled(
            "Toggle run options inline, then launch directly.",
            Style::default().fg(Color::DarkGray),
        ))
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(8),
            Constraint::Length(3),
        ])
        .split(inner_area(area));
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(32)])
        .split(chunks[1]);
    let selected_count = app.selected.iter().filter(|v| **v).count();
    let header = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                format!("Selected {} / {}", selected_count, app.raws.len()),
                Style::default().fg(accent()).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("Cursor {}", app.file_cursor + 1),
                Style::default().fg(Color::Gray),
            ),
        ]),
        header_subline,
    ])
    .block(panel("Selection", false));
    f.render_widget(header, chunks[0]);

    let start = app.file_cursor.saturating_sub(15);
    let end = (start + 30).min(app.raws.len());
    let mut items = Vec::new();
    for i in start..end {
        let mark = if app.selected[i] { "[x]" } else { "[ ]" };
        let line = format!("{mark} {}", app.raws[i].display());
        let style = if i == app.file_cursor {
            Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD)
        } else if app.selected[i] {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default()
        };
        items.push(ListItem::new(Line::from(Span::styled(line, style))));
    }
    let list = List::new(items).block(panel("RAW Files", true));
    f.render_widget(list, body[0]);

    let side = Paragraph::new(vec![
        Line::from(Span::styled(
            "Run Options",
            Style::default().fg(Color::Gray),
        )),
        Line::from(vec![
            Span::styled("[x] ", key_style()),
            Span::raw(format!(
                "XMP Sidecars: {}",
                if app.write_xmp { "ON" } else { "OFF" }
            )),
        ]),
        Line::from(vec![
            Span::styled("[v] ", key_style()),
            Span::raw(format!(
                "Previews: {}",
                if app.write_preview { "ON" } else { "OFF" }
            )),
        ]),
        Line::from(vec![
            Span::styled("[p] ", key_style()),
            Span::raw(format!("Preview Mode: {}", app.preview_mode.label())),
        ]),
        Line::from(Span::styled(
            "Final Crop Scale",
            Style::default().fg(Color::Gray),
        )),
        Line::from(vec![
            Span::styled(
                format!("{:+.2}%", app.final_crop_scale_pct),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("[0]", key_style()),
            Span::raw(" reset"),
        ]),
        Line::from(vec![
            Span::styled("[-]/[=] ", key_style()),
            Span::raw("+/-0.25%"),
        ]),
        Line::from(vec![
            Span::styled("[,]/[.] ", key_style()),
            Span::raw("+/-0.25%"),
        ]),
        Line::from(Span::styled(
            "Positive keeps more border.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("Preview Out: ", Style::default().fg(Color::Gray)),
            Span::styled(
                preview_dir(&args.out_dir).display().to_string(),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled("Confirm", Style::default().fg(Color::Gray))),
        Line::from("Press Enter to start"),
    ])
    .block(panel("Run Plan", false));
    f.render_widget(side, body[1]);

    let footer = Paragraph::new(Line::from(vec![
        Span::styled("Up/Down", key_style()),
        Span::raw(" move  "),
        Span::styled("Space", key_style()),
        Span::raw(" toggle  "),
        Span::styled("a", key_style()),
        Span::raw(" all  "),
        Span::styled("u", key_style()),
        Span::raw(" all-without-xmp  "),
        Span::styled("n", key_style()),
        Span::raw(" none  "),
        Span::styled("x", key_style()),
        Span::raw(" xmp  "),
        Span::styled("v", key_style()),
        Span::raw(" preview  "),
        Span::styled("p", key_style()),
        Span::raw(" preview-mode  "),
        Span::styled("[-]/[=]", key_style()),
        Span::raw(" crop-scale  "),
        Span::styled("[,]/[.]", key_style()),
        Span::raw(" step  "),
        Span::styled("0", key_style()),
        Span::raw(" reset  "),
        Span::styled("y/n", key_style()),
        Span::raw(" confirm overwrite  "),
        Span::styled("Enter", key_style()),
        Span::raw(" run"),
    ]))
    .block(panel("Keys", false));
    f.render_widget(footer, chunks[2]);
}

fn draw_running(f: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, app: &App) {
    draw_screen_chrome(f, area, "Croppy", "Running");
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(5),
            Constraint::Length(3),
        ])
        .split(inner_area(area));
    let (ok_count, failed_count, canceled_count) = summarize_results(&app.results);
    let pct = if app.progress_total == 0 {
        0
    } else {
        ((app.progress_done as f64 / app.progress_total as f64) * 100.0).round() as u16
    };
    let gauge_color = if app.cancel_requested {
        Color::Yellow
    } else {
        accent()
    };
    let gauge = Gauge::default()
        .block(panel("Progress", true))
        .gauge_style(Style::default().fg(gauge_color).bg(Color::Black))
        .label(format!(
            "{pct}%  ({}/{})",
            app.progress_done, app.progress_total
        ))
        .use_unicode(true)
        .percent(pct.min(100));
    f.render_widget(gauge, chunks[1]);

    let elapsed = app
        .run_started_at
        .map(|t| format_duration(t.elapsed()))
        .unwrap_or_else(|| "0s".to_string());
    let (pipeline_line, workers_line) = if let Some(stats) = &app.pipeline_stats {
        let pending = stats.pending_count.load(Ordering::Relaxed);
        let buffered = stats.buffered_count.load(Ordering::Relaxed);
        let buffered_peak = stats.buffered_peak.load(Ordering::Relaxed);
        let reading = stats.reading_count.load(Ordering::Relaxed);
        let reading_peak = stats.reading_peak.load(Ordering::Relaxed);
        let processing = stats.processing_count.load(Ordering::Relaxed);
        let processing_peak = stats.processing_peak.load(Ordering::Relaxed);
        (
            format!(
                "Pipeline: pending {pending} | buffered {buffered}/{} (peak {buffered_peak})",
                stats.buffered_capacity
            ),
            format!(
                "Workers: read {reading}/{} (peak {reading_peak}) | process {processing}/{} (peak {processing_peak})",
                stats.read_workers, stats.process_workers
            ),
        )
    } else {
        ("Pipeline: -".to_string(), "Workers: -".to_string())
    };
    let current = Paragraph::new(vec![
        Line::from(format!("Elapsed: {elapsed}")),
        Line::from(format!(
            "Done: {ok_count} ok  {failed_count} failed  {canceled_count} canceled"
        )),
        Line::from(pipeline_line),
        Line::from(workers_line),
        Line::from(vec![
            Span::styled("Current: ", Style::default().fg(Color::Gray)),
            Span::styled(
                if app.current_file.is_empty() {
                    "waiting"
                } else {
                    &app.current_file
                },
                Style::default().fg(Color::White),
            ),
        ]),
        if app.cancel_requested {
            Line::from(Span::styled(
                "Cancellation requested; finishing in-flight files...",
                Style::default().fg(Color::Yellow),
            ))
        } else {
            Line::from(Span::raw(""))
        },
    ])
    .block(panel("Status", false));
    f.render_widget(current, chunks[0]);

    let footer = Paragraph::new(Line::from(vec![
        Span::styled("c", key_style()),
        Span::raw(" cancel run  "),
        Span::styled("q", key_style()),
        Span::raw(" quit app"),
    ]))
    .block(panel("Keys", false));
    f.render_widget(footer, chunks[2]);
}

fn accent() -> Color {
    Color::White
}

fn key_style() -> Style {
    Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD)
}

fn preview_dir(out_dir: &Path) -> PathBuf {
    out_dir.join(PREVIEW_SUBDIR)
}

fn summarize_results(results: &[FileResult]) -> (usize, usize, usize) {
    let ok = results.iter().filter(|r| r.error.is_none()).count();
    let canceled = results.iter().filter(|r| is_canceled_result(r)).count();
    let failed = results
        .iter()
        .filter(|r| matches!(r.error.as_deref(), Some(msg) if msg != CANCELLED_MARKER))
        .count();
    (ok, failed, canceled)
}

fn is_canceled_result(result: &FileResult) -> bool {
    matches!(result.error.as_deref(), Some(CANCELLED_MARKER))
}

fn format_duration(dur: Duration) -> String {
    let secs = dur.as_secs();
    let mins = secs / 60;
    let rem = secs % 60;
    if mins > 0 {
        format!("{mins}m {rem}s")
    } else {
        format!("{rem}s")
    }
}

fn panel(title: &str, focused: bool) -> Block<'_> {
    let border_style = if focused {
        Style::default().add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Block::default()
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(Color::White),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style)
}

fn draw_screen_chrome(
    f: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    brand: &str,
    screen: &str,
) {
    let frame = Block::default()
        .title(Line::from(vec![
            Span::styled(
                format!(" {brand} "),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(screen, Style::default().fg(Color::DarkGray)),
        ]))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    f.render_widget(frame, area);
}

fn inner_area(area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    ratatui::layout::Rect {
        x: area.x.saturating_add(1),
        y: area.y.saturating_add(1),
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    }
}

fn adjust_final_crop_scale_pct(current: f32, delta: f32) -> f32 {
    clamp_final_crop_scale_pct(current + delta)
}

fn clamp_final_crop_scale_pct(value: f32) -> f32 {
    value.clamp(FINAL_CROP_SCALE_MIN_PCT, FINAL_CROP_SCALE_MAX_PCT)
}

fn handle_select_key(app: &mut App, args: &Args, code: KeyCode) -> Result<bool> {
    if app.overwrite_prompt_count.is_some() {
        match code {
            KeyCode::Char('q') => return Ok(true),
            KeyCode::Char('y') | KeyCode::Enter => {
                start_run(app, args, true)?;
            }
            KeyCode::Char('n') | KeyCode::Esc => {
                app.overwrite_prompt_count = None;
                app.select_notice = Some("Run canceled before overwrite.".to_string());
            }
            _ => {}
        }
        return Ok(false);
    }

    match code {
        KeyCode::Char('q') | KeyCode::Esc => return Ok(true),
        KeyCode::Up => app.file_cursor = app.file_cursor.saturating_sub(1),
        KeyCode::Down => app.file_cursor = (app.file_cursor + 1).min(app.raws.len() - 1),
        KeyCode::Char('x') => {
            app.write_xmp = !app.write_xmp;
            app.select_notice = None;
        }
        KeyCode::Char('v') => {
            app.write_preview = !app.write_preview;
            app.select_notice = None;
        }
        KeyCode::Char('p') => {
            app.preview_mode = app.preview_mode.next();
            app.select_notice = None;
        }
        KeyCode::Char('[') | KeyCode::Char('-') => {
            app.final_crop_scale_pct = adjust_final_crop_scale_pct(
                app.final_crop_scale_pct,
                -FINAL_CROP_SCALE_FINE_STEP_PCT,
            );
            app.select_notice = None;
        }
        KeyCode::Char(']') | KeyCode::Char('=') => {
            app.final_crop_scale_pct = adjust_final_crop_scale_pct(
                app.final_crop_scale_pct,
                FINAL_CROP_SCALE_FINE_STEP_PCT,
            );
            app.select_notice = None;
        }
        KeyCode::Char(',') => {
            app.final_crop_scale_pct = adjust_final_crop_scale_pct(
                app.final_crop_scale_pct,
                -FINAL_CROP_SCALE_FINE_STEP_PCT,
            );
            app.select_notice = None;
        }
        KeyCode::Char('.') => {
            app.final_crop_scale_pct = adjust_final_crop_scale_pct(
                app.final_crop_scale_pct,
                FINAL_CROP_SCALE_FINE_STEP_PCT,
            );
            app.select_notice = None;
        }
        KeyCode::Char('0') => {
            app.final_crop_scale_pct = 0.0;
            app.select_notice = None;
        }
        KeyCode::Char(' ') => {
            if let Some(v) = app.selected.get_mut(app.file_cursor) {
                *v = !*v;
            }
            app.select_notice = None;
        }
        KeyCode::Char('a') => {
            app.selected.fill(true);
            app.select_notice = None;
        }
        KeyCode::Char('n') => {
            app.selected.fill(false);
            app.select_notice = None;
        }
        KeyCode::Char('u') => {
            let mut selected_count = 0usize;
            for (idx, raw) in app.raws.iter().enumerate() {
                let keep = !raw.with_extension("xmp").exists();
                app.selected[idx] = keep;
                if keep {
                    selected_count += 1;
                }
            }
            app.select_notice = Some(format!(
                "Selected {selected_count} files that do not already have .xmp sidecars."
            ));
        }
        KeyCode::Enter => {
            start_run(app, args, false)?;
        }
        _ => {}
    }
    Ok(false)
}

fn start_run(app: &mut App, args: &Args, overwrite_confirmed: bool) -> Result<()> {
    let files = app.selected_paths();
    if files.is_empty() {
        app.select_notice = Some("Select at least one RAW file before running.".to_string());
        return Ok(());
    }
    if !app.write_xmp && !app.write_preview {
        app.select_notice =
            Some("Enable at least one output option (XMP sidecars or previews).".to_string());
        return Ok(());
    }
    if app.write_xmp && !overwrite_confirmed {
        let overwrite_count = files
            .iter()
            .filter(|raw| raw.with_extension("xmp").exists())
            .count();
        if overwrite_count > 0 {
            app.overwrite_prompt_count = Some(overwrite_count);
            app.select_notice = None;
            return Ok(());
        }
    }

    app.overwrite_prompt_count = None;
    app.select_notice = None;
    app.progress_total = files.len();
    app.progress_done = 0;
    app.current_file.clear();
    app.results.clear();
    app.cancel_requested = false;
    app.run_started_at = Some(Instant::now());
    let opts = RunOptions {
        write_xmp: app.write_xmp,
        write_preview: app.write_preview,
        preview_mode: app.preview_mode,
        max_edge: args.max_edge,
        out_dir: args.out_dir.clone(),
        final_crop_scale_pct: app.final_crop_scale_pct,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let (rx, stats) = spawn_worker(files, opts, cancel.clone())?;
    app.worker_rx = Some(rx);
    app.worker_cancel = Some(cancel);
    app.pipeline_stats = Some(stats);
    app.screen = Screen::Running;
    Ok(())
}

fn spawn_worker(
    files: Vec<PathBuf>,
    opts: RunOptions,
    cancel: Arc<AtomicBool>,
) -> Result<(Receiver<WorkerMsg>, Arc<PipelineStats>)> {
    fs::create_dir_all(preview_dir(&opts.out_dir))?;
    let (tx, rx) = mpsc::channel();
    let cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Keep a meaningful reader stage on high-core systems so processors are not starved.
    let read_workers = (cores / 3).clamp(2, 8);
    let process_workers = cores.saturating_sub(read_workers).max(2);
    let buffered_jobs = cores.max(1);
    let stats = Arc::new(PipelineStats::new(
        files.len(),
        buffered_jobs,
        read_workers,
        process_workers,
    ));
    let stats_for_worker = Arc::clone(&stats);
    thread::spawn(move || {
        let buffered_jobs = stats_for_worker.buffered_capacity;

        let queue = Arc::new(Mutex::new(VecDeque::from(files)));
        let (raw_tx, raw_rx) = mpsc::sync_channel::<RawJob>(buffered_jobs);
        let shared_raw_rx = Arc::new(Mutex::new(raw_rx));

        let mut process_handles = Vec::with_capacity(process_workers);
        for _ in 0..process_workers {
            let opts_cloned = opts.clone();
            let tx_cloned = tx.clone();
            let cancel_cloned = Arc::clone(&cancel);
            let raw_rx_cloned = Arc::clone(&shared_raw_rx);
            let stats_cloned = Arc::clone(&stats_for_worker);
            process_handles.push(thread::spawn(move || {
                loop {
                    let next_job = {
                        let rx_guard = raw_rx_cloned.lock().expect("raw_rx mutex poisoned");
                        rx_guard.recv()
                    };
                    let Ok(job) = next_job else {
                        break;
                    };
                    stats_cloned.dec_buffered();
                    stats_cloned.start_processing();
                    let res = process_raw_job(job, &opts_cloned, &tx_cloned, &cancel_cloned);
                    stats_cloned.finish_processing();
                    let _ = tx_cloned.send(WorkerMsg::Finished(res));
                }
            }));
        }

        let mut read_handles = Vec::with_capacity(read_workers);
        for _ in 0..read_workers {
            let queue_cloned = Arc::clone(&queue);
            let raw_tx_cloned = raw_tx.clone();
            let tx_cloned = tx.clone();
            let cancel_cloned = Arc::clone(&cancel);
            let stats_cloned = Arc::clone(&stats_for_worker);
            read_handles.push(thread::spawn(move || {
                loop {
                    let maybe_raw = {
                        let mut guard = queue_cloned.lock().expect("queue mutex poisoned");
                        guard.pop_front()
                    };
                    let Some(raw) = maybe_raw else {
                        break;
                    };
                    stats_cloned.mark_path_taken();

                    if cancel_cloned.load(Ordering::Relaxed) {
                        let _ = tx_cloned.send(WorkerMsg::Finished(canceled_result()));
                        continue;
                    }

                    let _ = tx_cloned.send(WorkerMsg::Started(format!("queue: {}", raw.display())));
                    stats_cloned.start_reading();
                    stats_cloned.inc_buffered();
                    let send_res = raw_tx_cloned.send(RawJob { raw });
                    stats_cloned.finish_reading();
                    if send_res.is_err() {
                        stats_cloned.dec_buffered();
                        break;
                    }
                }
            }));
        }
        drop(raw_tx);

        for handle in read_handles {
            let _ = handle.join();
        }
        for handle in process_handles {
            let _ = handle.join();
        }

        let _ = tx.send(WorkerMsg::Done);
    });
    Ok((rx, stats))
}

fn process_raw_job(
    job: RawJob,
    opts: &RunOptions,
    tx: &Sender<WorkerMsg>,
    cancel: &AtomicBool,
) -> FileResult {
    let RawJob { raw } = job;
    if cancel.load(Ordering::Relaxed) {
        return canceled_result();
    }
    let _ = tx.send(WorkerMsg::Started(format!("process: {}", raw.display())));

    let mut out = FileResult {
        preview: None,
        xmp: None,
        error: None,
    };

    let res = (|| -> Result<()> {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!(CANCELLED_MARKER));
        }
        let decoded = decode_raw_to_rgb_with_hint(&raw, opts.max_edge)?;
        if let Some(warn) = decoded.warning {
            let _ = tx.send(WorkerMsg::Started(format!("warning: {warn}")));
        }
        let rgb_full = decoded.image;
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!(CANCELLED_MARKER));
        }
        let rgb = resize_rgb_max_edge(&rgb_full, opts.max_edge);
        let gray = rgb_to_gray(&rgb);
        let preprocess_cfg = PreprocessConfig {
            invert: true,
            flip_180: true,
            black_pct: 2.0,
            white_pct: 80.0,
            knee_pct: 2.0,
        };
        let prepared = prepare_image(gray, preprocess_cfg);
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!(CANCELLED_MARKER));
        }
        let prep_gray = rgb_to_gray(&prepared);
        let refine = run_detection_with_rotation_refine(
            &prep_gray,
            RotationRefineConfig {
                refine_rotation: true,
                apply_rotation_decision: true,
                max_refine_abs_deg: 3.0,
            },
        )
        .ok_or_else(|| anyhow!("boundary detection failed"))?;

        if opts.write_preview {
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!(CANCELLED_MARKER));
            }
            let preview_path = preview_path_for(&raw, &opts.out_dir);
            guard_write_target(&raw, &preview_path, "preview")?;
            match opts.preview_mode {
                PreviewMode::DebugOverlay => write_overlay_preview(
                    &prepared,
                    &refine,
                    opts.final_crop_scale_pct,
                    &preview_path,
                )?,
                PreviewMode::FinalCrop => write_final_crop_preview(
                    &prepared,
                    &refine,
                    opts.final_crop_scale_pct,
                    &preview_path,
                )?,
                PreviewMode::FinalCropFramed => write_framed_final_crop_preview(
                    &prepared,
                    &refine,
                    opts.final_crop_scale_pct,
                    &preview_path,
                )?,
            }
            out.preview = Some(preview_path);
        }

        if opts.write_xmp {
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!(CANCELLED_MARKER));
            }
            let xmp_crop = xmp_crop_from_detection(
                refine.detection.inner,
                refine.rotation_applied_deg,
                preprocess_cfg.flip_180,
                opts.final_crop_scale_pct,
            );
            let xmp_path = raw.with_extension("xmp");
            guard_write_target(&raw, &xmp_path, "xmp sidecar")?;
            write_xmp_sidecar(
                &xmp_path,
                xmp_crop.left,
                xmp_crop.top,
                xmp_crop.right,
                xmp_crop.bottom,
                xmp_crop.angle_deg,
            )?;
            out.xmp = Some(xmp_path);
        }
        Ok(())
    })();

    if let Err(e) = res {
        if e.to_string() == CANCELLED_MARKER {
            out = canceled_result();
        } else {
            out.error = Some(e.to_string());
        }
    }
    out
}

fn preview_path_for(raw: &Path, out_dir: &Path) -> PathBuf {
    let mut base = raw
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("preview")
        .to_string();
    base.push_str(".jpg");
    preview_dir(out_dir).join(base)
}

fn write_overlay_preview(
    prepared: &image::RgbImage,
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
    preview_path: &Path,
) -> Result<()> {
    let mut overlay = prepared.clone();
    draw_norm_rect(
        &mut overlay,
        refine.detection_initial.inner,
        image::Rgb([255, 255, 0]),
    );
    if let (Some(refined_det), Some(applied_deg)) =
        (refine.detection_refined, refine.rotation_applied_deg)
    {
        draw_refined_inner_backproject(
            &mut overlay,
            refined_det.inner,
            applied_deg.to_radians(),
            image::Rgb([80, 255, 255]),
        );
    }
    if final_crop_scale_pct.abs() > f32::EPSILON {
        let (final_inner, theta_opt) = final_preview_inner_bounds(refine, final_crop_scale_pct);
        if let Some(theta) = theta_opt {
            draw_refined_inner_backproject(
                &mut overlay,
                final_inner,
                theta,
                image::Rgb([120, 255, 120]),
            );
        } else {
            draw_norm_rect(&mut overlay, final_inner, image::Rgb([120, 255, 120]));
        }
    }
    overlay.save_with_format(preview_path, image::ImageFormat::Jpeg)?;
    Ok(())
}

fn write_final_crop_preview(
    prepared: &image::RgbImage,
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
    preview_path: &Path,
) -> Result<()> {
    let crop = render_final_crop_image(prepared, refine, final_crop_scale_pct);
    crop.save_with_format(preview_path, image::ImageFormat::Jpeg)?;
    Ok(())
}

fn write_framed_final_crop_preview(
    prepared: &image::RgbImage,
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
    preview_path: &Path,
) -> Result<()> {
    let crop = render_final_crop_image(prepared, refine, final_crop_scale_pct);
    let framed = render_crop_with_white_frame(&crop);
    framed.save_with_format(preview_path, image::ImageFormat::Jpeg)?;
    Ok(())
}

fn final_preview_inner_bounds(
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
) -> (BoundsNorm, Option<f32>) {
    if let (Some(refined_det), Some(applied_deg)) =
        (refine.detection_refined, refine.rotation_applied_deg)
    {
        let scaled =
            scale_bounds_about_center(normalize_bounds(refined_det.inner), final_crop_scale_pct);
        (scaled, Some(applied_deg.to_radians()))
    } else {
        let scaled = scale_bounds_about_center(
            normalize_bounds(refine.detection.inner),
            final_crop_scale_pct,
        );
        (scaled, None)
    }
}

fn render_final_crop_image(
    prepared: &image::RgbImage,
    refine: &DetectRefineRun,
    final_crop_scale_pct: f32,
) -> image::RgbImage {
    let (final_inner, theta_opt) = final_preview_inner_bounds(refine, final_crop_scale_pct);
    if let Some(theta) = theta_opt {
        let rotated = rotate_rgb_about_center(prepared, theta);
        extract_norm_crop(&rotated, final_inner)
    } else {
        extract_norm_crop(prepared, final_inner)
    }
}

fn extract_norm_crop(img: &image::RgbImage, b: BoundsNorm) -> image::RgbImage {
    let w = img.width() as f32;
    let h = img.height() as f32;
    let x1 = (b.left * w).round().clamp(0.0, w - 1.0) as u32;
    let x2 = (b.right * w).round().clamp(0.0, w) as u32;
    let y1 = (b.top * h).round().clamp(0.0, h - 1.0) as u32;
    let y2 = (b.bottom * h).round().clamp(0.0, h) as u32;
    let cw = x2.saturating_sub(x1).max(1);
    let ch = y2.saturating_sub(y1).max(1);
    image::imageops::crop_imm(img, x1, y1, cw, ch).to_image()
}

fn render_crop_with_white_frame(crop: &image::RgbImage) -> image::RgbImage {
    let pad_x = ((crop.width() as f32 * FRAME_PAD_FRAC).round() as u32).max(FRAME_PAD_MIN_PX);
    let pad_y = ((crop.height() as f32 * FRAME_PAD_FRAC).round() as u32).max(FRAME_PAD_MIN_PX);
    let out_w = crop.width().saturating_add(pad_x.saturating_mul(2)).max(1);
    let out_h = crop.height().saturating_add(pad_y.saturating_mul(2)).max(1);
    let mut framed = image::RgbImage::from_pixel(out_w, out_h, image::Rgb([255, 255, 255]));

    for (x, y, px) in crop.enumerate_pixels() {
        framed.put_pixel(x + pad_x, y + pad_y, *px);
    }

    framed
}

fn canceled_result() -> FileResult {
    FileResult {
        preview: None,
        xmp: None,
        error: Some(CANCELLED_MARKER.to_string()),
    }
}

fn xmp_crop_from_detection(
    inner_detected: BoundsNorm,
    rotation_applied_deg: Option<f32>,
    preprocess_flip_180: bool,
    final_crop_scale_pct: f32,
) -> XmpCrop {
    let mut bounds = inner_detected;
    if preprocess_flip_180 {
        // Detection runs in a frame rotated 180 degrees; mirror bounds back so
        // sidecar geometry matches RAW orientation. CropAngle sign conversion is
        // handled separately below.
        bounds = rotate_bounds_180(bounds);
    }
    bounds = normalize_bounds(bounds);
    bounds = scale_bounds_about_center(bounds, final_crop_scale_pct);
    XmpCrop {
        left: bounds.left,
        top: bounds.top,
        right: bounds.right,
        bottom: bounds.bottom,
        // Internal refine rotates image pixels by `rotation_applied_deg`, while
        // Lightroom's CropAngle uses the opposite sign convention.
        angle_deg: -rotation_applied_deg.unwrap_or(0.0),
    }
}

fn rotate_bounds_180(b: BoundsNorm) -> BoundsNorm {
    BoundsNorm {
        left: 1.0 - b.right,
        top: 1.0 - b.bottom,
        right: 1.0 - b.left,
        bottom: 1.0 - b.top,
    }
}

fn normalize_bounds(b: BoundsNorm) -> BoundsNorm {
    let mut left = b.left.clamp(0.0, 1.0);
    let mut right = b.right.clamp(0.0, 1.0);
    let mut top = b.top.clamp(0.0, 1.0);
    let mut bottom = b.bottom.clamp(0.0, 1.0);
    if left > right {
        std::mem::swap(&mut left, &mut right);
    }
    if top > bottom {
        std::mem::swap(&mut top, &mut bottom);
    }
    BoundsNorm {
        left,
        top,
        right,
        bottom,
    }
}

fn scale_bounds_about_center(bounds: BoundsNorm, scale_pct: f32) -> BoundsNorm {
    let scale_pct = clamp_final_crop_scale_pct(scale_pct);
    let scale = (1.0 + (scale_pct / 100.0)).max(0.01);

    let cx = (bounds.left + bounds.right) * 0.5;
    let cy = (bounds.top + bounds.bottom) * 0.5;
    let half_w = (bounds.right - bounds.left).abs() * 0.5 * scale;
    let half_h = (bounds.bottom - bounds.top).abs() * 0.5 * scale;

    normalize_bounds(BoundsNorm {
        left: cx - half_w,
        top: cy - half_h,
        right: cx + half_w,
        bottom: cy + half_h,
    })
}

fn guard_write_target(raw: &Path, target: &Path, output_kind: &str) -> Result<()> {
    if raw == target {
        return Err(anyhow!(
            "refusing to write {output_kind}: target path equals source RAW path ({})",
            raw.display()
        ));
    }
    if is_supported_raw(target) {
        return Err(anyhow!(
            "refusing to write {output_kind}: target has RAW extension ({})",
            target.display()
        ));
    }

    let raw_abs = fs::canonicalize(raw).unwrap_or_else(|_| raw.to_path_buf());
    let target_abs = fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
    if raw_abs == target_abs {
        return Err(anyhow!(
            "refusing to write {output_kind}: target resolves to source RAW path ({})",
            target.display()
        ));
    }

    if let (Ok(raw_meta), Ok(target_meta)) = (fs::metadata(raw), fs::metadata(target)) {
        #[cfg(unix)]
        {
            if raw_meta.dev() == target_meta.dev() && raw_meta.ino() == target_meta.ino() {
                return Err(anyhow!(
                    "refusing to write {output_kind}: target points to same file inode as source RAW ({})",
                    target.display()
                ));
            }
        }
    }
    Ok(())
}

fn write_xmp_sidecar(
    path: &Path,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    angle_deg: f32,
) -> Result<()> {
    let xmp = format!(
        concat!(
            "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n",
            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n",
            " <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n",
            "  <rdf:Description rdf:about=\"\"\n",
            "    xmlns:crs=\"http://ns.adobe.com/camera-raw-settings/1.0/\"\n",
            "    crs:HasCrop=\"True\"\n",
            "    crs:CropConstrainAspectRatio=\"True\"\n",
            "    crs:CropLeft=\"{left:.6}\"\n",
            "    crs:CropTop=\"{top:.6}\"\n",
            "    crs:CropRight=\"{right:.6}\"\n",
            "    crs:CropBottom=\"{bottom:.6}\"\n",
            "    crs:CropAngle=\"{angle:.6}\" />\n",
            " </rdf:RDF>\n",
            "</x:xmpmeta>\n",
            "<?xpacket end=\"w\"?>\n"
        ),
        left = left,
        top = top,
        right = right,
        bottom = bottom,
        angle = angle_deg
    );
    fs::write(path, xmp)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BoundsNorm, PreviewMode, xmp_crop_from_detection};

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn xmp_crop_keeps_bounds_without_flip() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.1,
                top: 0.2,
                right: 0.8,
                bottom: 0.9,
            },
            Some(-0.5),
            false,
            0.0,
        );
        assert_close(out.left, 0.1);
        assert_close(out.top, 0.2);
        assert_close(out.right, 0.8);
        assert_close(out.bottom, 0.9);
        assert_close(out.angle_deg, 0.5);
    }

    #[test]
    fn xmp_crop_mirrors_bounds_when_preprocess_flipped() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.1,
                top: 0.2,
                right: 0.8,
                bottom: 0.9,
            },
            Some(1.25),
            true,
            0.0,
        );
        assert_close(out.left, 0.2);
        assert_close(out.top, 0.1);
        assert_close(out.right, 0.9);
        assert_close(out.bottom, 0.8);
        assert_close(out.angle_deg, -1.25);
    }

    #[test]
    fn xmp_crop_clamps_and_normalizes_bounds() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 1.2,
                top: 0.9,
                right: -0.1,
                bottom: 0.2,
            },
            None,
            false,
            0.0,
        );
        assert_close(out.left, 0.0);
        assert_close(out.top, 0.2);
        assert_close(out.right, 1.0);
        assert_close(out.bottom, 0.9);
        assert_close(out.angle_deg, 0.0);
    }

    #[test]
    fn xmp_crop_expands_around_center() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.2,
                top: 0.3,
                right: 0.8,
                bottom: 0.7,
            },
            None,
            false,
            10.0,
        );
        assert_close(out.left, 0.17);
        assert_close(out.top, 0.28);
        assert_close(out.right, 0.83);
        assert_close(out.bottom, 0.72);
        assert_close(out.angle_deg, 0.0);
    }

    #[test]
    fn xmp_crop_shrinks_around_center() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.2,
                top: 0.3,
                right: 0.8,
                bottom: 0.7,
            },
            None,
            false,
            -10.0,
        );
        assert_close(out.left, 0.23);
        assert_close(out.top, 0.32);
        assert_close(out.right, 0.77);
        assert_close(out.bottom, 0.68);
        assert_close(out.angle_deg, 0.0);
    }

    #[test]
    fn xmp_crop_scale_is_clamped() {
        let out = xmp_crop_from_detection(
            BoundsNorm {
                left: 0.2,
                top: 0.3,
                right: 0.8,
                bottom: 0.7,
            },
            None,
            false,
            100.0,
        );
        assert_close(out.left, 0.14);
        assert_close(out.top, 0.26);
        assert_close(out.right, 0.86);
        assert_close(out.bottom, 0.74);
    }

    #[test]
    fn preview_mode_cycles() {
        assert_eq!(PreviewMode::DebugOverlay.next(), PreviewMode::FinalCrop);
        assert_eq!(PreviewMode::FinalCrop.next(), PreviewMode::FinalCropFramed);
        assert_eq!(
            PreviewMode::FinalCropFramed.next(),
            PreviewMode::DebugOverlay
        );
    }

    #[test]
    fn framed_crop_adds_white_border_area() {
        let crop = image::RgbImage::from_pixel(100, 50, image::Rgb([12, 34, 56]));
        let framed = super::render_crop_with_white_frame(&crop);
        let pad_x = ((crop.width() as f32 * super::FRAME_PAD_FRAC).round() as u32)
            .max(super::FRAME_PAD_MIN_PX);
        let pad_y = ((crop.height() as f32 * super::FRAME_PAD_FRAC).round() as u32)
            .max(super::FRAME_PAD_MIN_PX);
        assert!(framed.width() > crop.width());
        assert!(framed.height() > crop.height());
        // Top-left remains background white.
        assert_eq!(*framed.get_pixel(0, 0), image::Rgb([255, 255, 255]));
        // The crop is pasted directly with no synthetic border stroke.
        assert_eq!(*framed.get_pixel(pad_x, pad_y), image::Rgb([12, 34, 56]));
    }
}
