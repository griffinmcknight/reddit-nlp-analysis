#!/bin/bash

# Navigate to the src directory
cd src

# Create the new scrapers directory
mkdir -p scrapers

# Move data scraping scripts to the scrapers directory
mv data/reddit_scraper.py scrapers/
mv data/scrape_flagged_user_activity.py scrapers/

# Leave only generated data in the data directory
# (already in data directory, so no need to move flagged_users.txt)

# Print a message to confirm completion
echo "Project structure updated successfully!"
