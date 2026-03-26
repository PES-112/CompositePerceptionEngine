#!/bin/bash
# setup_gcs.sh — Upload SANPO to Google Cloud Storage

# ============================================================================
# OPTION 1: Upload SANPO from your MacBook to GCS
# ============================================================================

# Set your bucket name
BUCKET_NAME="your-sanpo-bucket"
SANPO_LOCAL="/path/to/sanpo"

echo "Setting up GCS bucket: gs://$BUCKET_NAME"

# 1. Create bucket (if not exists)
gsutil mb -c STANDARD -l us-central1 gs://$BUCKET_NAME/ 2>/dev/null || echo "Bucket exists"

# 2. Upload SANPO data (this will take a while!)
echo "Uploading SANPO data..."
gsutil -m cp -r $SANPO_LOCAL/* gs://$BUCKET_NAME/sanpo/

# 3. Verify upload
echo "Verifying structure..."
gsutil ls -r gs://$BUCKET_NAME/sanpo/ | head -20

echo ""
echo "✓ Upload complete!"
echo ""
echo "Your GCS path is:"
echo "  gs://$BUCKET_NAME/sanpo"
echo ""
echo "Use this in Colab as:"
echo "  GCS_SANPO_PATH = 'gs://$BUCKET_NAME/sanpo'"
