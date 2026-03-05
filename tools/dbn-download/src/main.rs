//! CLI tool to download historical market data from Databento.
//!
//! Requires a `DATABENTO_API_KEY` environment variable.
//!
//! Usage:
//!   dbn-download --symbol ESH25 --start 2025-01-06 --end 2025-01-06 -o data/es_20250106.dbn.zst

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use databento::dbn::decode::DbnMetadata;
use databento::dbn::{SType, Schema};
use databento::historical::timeseries::GetRangeToFileParams;
use databento::HistoricalClient;
use time::macros::format_description;

#[derive(Parser, Debug)]
#[command(name = "dbn-download")]
#[command(about = "Download historical Databento MBP-10 data for paper trading")]
struct Args {
    /// CME futures symbol (e.g. "ESH25", "ESM25", "ES.FUT" for continuous)
    #[arg(short, long)]
    symbol: String,

    /// Start date in YYYY-MM-DD format
    #[arg(long)]
    start: String,

    /// End date in YYYY-MM-DD format
    #[arg(long)]
    end: String,

    /// Output file path (default: data/<symbol>_<start>.dbn.zst)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Dataset (default: GLBX.MDP3 for CME)
    #[arg(long, default_value = "GLBX.MDP3")]
    dataset: String,

    /// Schema (default: mbp-10)
    #[arg(long, default_value = "mbp-10")]
    schema: String,

    /// Symbol type (default: raw_symbol)
    #[arg(long, default_value = "raw_symbol")]
    stype: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let format = format_description!("[year]-[month]-[day]");
    let start_date = time::Date::parse(&args.start, &format)?;
    let end_date = time::Date::parse(&args.end, &format)?;

    let start_dt = start_date.with_hms(0, 0, 0)?.assume_utc();
    let end_dt = end_date.with_hms(23, 59, 59)?.assume_utc();

    let output = args.output.unwrap_or_else(|| {
        let filename = format!("{}_{}.dbn.zst", args.symbol, args.start);
        PathBuf::from("data").join(filename)
    });

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let schema: Schema = match args.schema.as_str() {
        "mbp-10" => Schema::Mbp10,
        "mbp-1" => Schema::Mbp1,
        "trades" => Schema::Trades,
        other => anyhow::bail!("Unsupported schema: {other}"),
    };

    let stype: SType = match args.stype.as_str() {
        "raw_symbol" => SType::RawSymbol,
        "parent" => SType::Parent,
        "continuous" => SType::Continuous,
        other => anyhow::bail!("Unsupported stype: {other}"),
    };

    println!("Downloading {schema:?} data for {} ...", args.symbol);
    println!("  Dataset:  {}", args.dataset);
    println!("  Range:    {} to {}", args.start, args.end);
    println!("  Output:   {}", output.display());
    println!();

    let mut client = HistoricalClient::builder().key_from_env()?.build()?;

    let params = GetRangeToFileParams::builder()
        .dataset(args.dataset)
        .symbols(args.symbol.clone())
        .schema(schema)
        .date_time_range((start_dt)..(end_dt))
        .stype_in(stype)
        .path(output.clone())
        .build();

    let decoder = client
        .timeseries()
        .get_range_to_file(&params)
        .await?;

    let metadata = decoder.metadata();
    println!("Download complete!");
    println!("  Schema:      {:?}", metadata.schema);
    println!("  Start:       {:?}", metadata.start);
    println!("  End:         {:?}", metadata.end);
    println!("  Symbols:     {:?}", metadata.symbols);
    println!("  File:        {}", output.display());

    Ok(())
}
