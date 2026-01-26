#!/usr/bin/env python3
"""
Video Downloader Script
Downloads videos from CSV file based on engagementId and videoS3Url columns.
"""

import csv
import os
import sys
import requests
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import logging
from typing import Set, Dict, Optional
import argparse


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VideoDownloader:
    """Downloads videos from CSV data with error handling and progress tracking."""
    
    def __init__(self, csv_path: str, output_folder: str):
        """
        Initialize the video downloader.
        
        Args:
            csv_path: Path to the CSV file
            output_folder: Folder where videos will be saved
        """
        self.csv_path = csv_path
        self.output_folder = Path(output_folder)
        self.seen_engagement_ids: Set[str] = set()
        self.download_stats = {
            'success': 0,
            'skipped_existing': 0,
            'skipped_duplicate': 0,
            'error_na_url': 0,
            'error_download': 0,
            'error_invalid_url': 0
        }
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def get_extension_from_url(self, url: str) -> str:
        """
        Extract file extension from URL.
        
        Args:
            url: The video URL
            
        Returns:
            File extension (including dot) or '.mp4' as default
        """
        parsed = urlparse(url)
        path = parsed.path
        _, ext = os.path.splitext(path)
        return ext if ext else '.mp4'
    
    def download_video(self, url: str, output_path: Path, retries: int = 1) -> bool:
        """
        Download a video from URL with progress bar and retry logic.
        
        Args:
            url: Video URL to download
            output_path: Where to save the video
            retries: Number of retry attempts (0 or 1)
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for {output_path.name}")
                
                # Stream the download
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Get total file size
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar
                with open(output_path, 'wb') as f, tqdm(
                    desc=f"Downloading {output_path.name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                logger.info(f"Successfully downloaded: {output_path.name}")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Download error for {output_path.name}: {e}")
                # Delete partial file if it exists
                if output_path.exists():
                    output_path.unlink()
                
                if attempt == retries:
                    return False
        
        return False
    
    def process_row(self, row: Dict[str, str]) -> bool:
        """
        Process a single CSV row and download the video.
        
        Args:
            row: Dictionary containing the CSV row data
            
        Returns:
            True if video was downloaded, False otherwise
        """
        engagement_id = row.get('engagementId', '').strip()
        video_url = row.get('videoS3Url', '').strip()
        
        # Check if engagementId is empty
        if not engagement_id:
            logger.warning("Empty engagementId found, skipping row")
            return False
        
        # Check for duplicate engagementId
        if engagement_id in self.seen_engagement_ids:
            logger.info(f"Duplicate engagementId found: {engagement_id}, skipping (first occurrence already processed)")
            self.download_stats['skipped_duplicate'] += 1
            return False
        
        # Mark as seen
        self.seen_engagement_ids.add(engagement_id)
        
        # Check for NA or empty URL
        if not video_url or video_url.upper() == 'NA':
            logger.error(f"NA or empty videoS3Url for engagementId: {engagement_id}")
            self.download_stats['error_na_url'] += 1
            return False
        
        # Validate URL format
        try:
            parsed = urlparse(video_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")
        except Exception as e:
            logger.error(f"Invalid URL format for engagementId {engagement_id}: {video_url} - {e}")
            self.download_stats['error_invalid_url'] += 1
            return False
        
        # Get file extension
        ext = self.get_extension_from_url(video_url)
        output_path = self.output_folder / f"{engagement_id}{ext}"
        
        # Check if file already exists
        if output_path.exists():
            logger.info(f"Video already exists, skipping: {output_path.name}")
            self.download_stats['skipped_existing'] += 1
            return False
        
        # Download the video
        if self.download_video(video_url, output_path, retries=1):
            self.download_stats['success'] += 1
            return True
        else:
            self.download_stats['error_download'] += 1
            return False
    
    def download_range(self, sno_from: int, sno_till: int) -> None:
        """
        Download videos for a specific SNo range.
        
        Args:
            sno_from: Starting SNo (inclusive)
            sno_till: Ending SNo (inclusive)
        """
        logger.info(f"Starting download process for SNo {sno_from} to {sno_till}")
        logger.info(f"CSV file: {self.csv_path}")
        logger.info(f"Output folder: {self.output_folder}")
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        sno = int(row.get('SNo', '0'))
                        
                        # Check if SNo is in range
                        if sno < sno_from or sno > sno_till:
                            continue
                        
                        logger.info(f"\n--- Processing SNo: {sno} ---")
                        self.process_row(row)
                        
                    except ValueError as e:
                        logger.warning(f"Invalid SNo value in row: {e}")
                        continue
        
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print download statistics summary."""
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*60)
        logger.info(f"Successfully downloaded: {self.download_stats['success']}")
        logger.info(f"Skipped (already exists): {self.download_stats['skipped_existing']}")
        logger.info(f"Skipped (duplicate engagementId): {self.download_stats['skipped_duplicate']}")
        logger.info(f"Errors (NA/empty URL): {self.download_stats['error_na_url']}")
        logger.info(f"Errors (invalid URL): {self.download_stats['error_invalid_url']}")
        logger.info(f"Errors (download failed): {self.download_stats['error_download']}")
        logger.info("="*60)


def main():
    """Main function to parse arguments and run the downloader."""
    parser = argparse.ArgumentParser(
        description='Download videos from CSV file based on SNo range'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='Sample/AI Optom Co-Pilot - Dataset + Trackr - Consolidated 800.csv',
        help='Path to CSV file (default: Sample/AI Optom Co-Pilot - Dataset + Trackr - Consolidated 800.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Sample/videos-2',
        help='Output folder for videos (default: Sample/videos-2)'
    )
    parser.add_argument(
        '--from',
        dest='sno_from',
        type=int,
        required=True,
        help='Starting SNo (inclusive)'
    )
    parser.add_argument(
        '--till',
        dest='sno_till',
        type=int,
        required=True,
        help='Ending SNo (inclusive)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.sno_from < 1:
        logger.error("SNo 'from' must be >= 1")
        sys.exit(1)
    
    if args.sno_till < args.sno_from:
        logger.error("SNo 'till' must be >= SNo 'from'")
        sys.exit(1)
    
    # Create downloader and run
    downloader = VideoDownloader(args.csv, args.output)
    
    try:
        downloader.download_range(args.sno_from, args.sno_till)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
