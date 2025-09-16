#!/usr/bin/env python3
"""
Standalone BAM parser for extracting polyA tail length estimates
from Dorado BAM files.

This script extracts the 'pt' (polyA tail) tags from Dorado BAM
files and export them to CSV format for further analysis with corridora.
"""

import warnings
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import pysam

warnings.filterwarnings("ignore")


def parse_dorado_bam(bam_file: str) -> pd.DataFrame:
    """
    Parse Dorado BAM file and extract polyA tail length estimates.

    Args:
        bam_file: Path to the Dorado BAM file

    Returns:
        DataFrame with columns: read_id, polya_len
    """
    estimated_lengths = []
    total_reads = 0
    valid_reads = 0

    with pysam.AlignmentFile(bam_file, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            total_reads += 1

            # Extract polyA tail length from 'pt' tag
            pt_value = dict(read.tags).get("pt")

            if pt_value is not None:
                estimated_lengths.append(
                    {"read_id": read.query_name, "polya_len": pt_value}
                )
                if pt_value > 0:
                    valid_reads += 1

    click.echo(
        (
            f"Processed {total_reads:,} reads, found {valid_reads:,} ",
            f"with polyA estimates ({valid_reads / total_reads * 100:.2f}%)",
        )
    )

    return pd.DataFrame(estimated_lengths)


@click.command()
@click.argument("bam_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output CSV file (default: based on input filename)",
)
@click.option(
    "--sample-name",
    "-s",
    help="Sample name to add as a column (default: filename)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
@click.version_option()
def main(
    bam_file: str,
    output: Optional[str],
    sample_name: Optional[str],
    quiet: bool,
):
    """
    Extract polyA tail length estimates from Dorado BAM files.

    BAM_FILE: Path to the Dorado BAM file containing polyA estimates.

    This tool extracts the 'pt' (polyA tail) tags from reads and exports
    them to CSV format for analysis with corridora or other tools.

    Example usage:

        python parse_bam_polya.py sample.bam -o sample_polya.csv
    """
    if not quiet:
        click.echo("Corridora BAM PolyA Parser")
        click.echo("=" * 40)

    if output is None:
        bam_path = Path(bam_file)
        output = bam_path.with_suffix(".polya.csv")

    if sample_name is None:
        sample_name = Path(bam_file).stem

    if not quiet:
        click.echo(f"Input BAM: {bam_file}")
        click.echo(f"Output CSV: {output}")
        click.echo(f"Sample name: {sample_name}")

    try:
        if not quiet:
            click.echo("\nParsing BAM file...")

        df = parse_dorado_bam(bam_file)

        if len(df) == 0:
            click.echo("No polyA estimates found in BAM file")
            click.echo(
                (
                    "   Make sure this is a Dorado BAM from basecalling ",
                    "with polyA estimation enabled",
                )
            )
            return

        df["sample_id"] = sample_name
        df = df[["sample_id", "read_id", "polya_len"]]
        df.to_csv(output, index=False)
        click.echo(f"Results saved to: {output}")

    except Exception as e:
        click.echo(f"Error processing BAM file: {e}", err=True)
        return 1


if __name__ == "__main__":
    main()
