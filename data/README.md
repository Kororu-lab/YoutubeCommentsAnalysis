# Data Directory

## 📁 Directory Structure

```
data/
├── README.md                    # This file
├── sample_youtube_comments.csv  # Sample data for testing
├── example_data_format.csv      # Data format example
└── .gitkeep                     # Keep directory in Git
```

## 📊 Data Format Requirements

Your CSV files should contain the following columns:

### Required Columns
| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `text` or `comment_text` | string | Comment content | "정말 충격적인 뉴스네요..." |
| `date` | datetime | Comment date | "2023-01-15 14:30:25" |

### Optional Columns
| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `upvotes` | int | Number of likes | 15 |
| `downvotes` | int | Number of dislikes | 2 |
| `author` | string | Comment author | "user123" |
| `video_title` | string | Video title | "뉴스 제목" |
| `video_id` | string | YouTube video ID | "dQw4w9WgXcQ" |

## 🔧 How to Use Your Own Data

1. **Prepare your CSV file** with the required columns
2. **Place it in this directory** (`data/`)
3. **Update the configuration** in `config.py`:
   ```python
   DATA_FILES = {
       'youtube_comments': os.path.join(DATA_DIR, 'your_data.csv'),
       # ...
   }
   ```

## 📝 Sample Data

The `sample_youtube_comments.csv` file contains anonymized sample data that demonstrates the expected format. This data is safe to commit to Git and helps new users understand the structure.

## 🔒 Data Privacy

- **Real data files** (*.csv, *.xlsx, *.json) are automatically ignored by Git
- Only sample/example files (sample_*.csv, example_*.csv) are tracked
- Make sure your real data files don't start with "sample_" or "example_"

## 📋 Data Collection Guidelines

When collecting YouTube comments data:

1. **Respect YouTube's Terms of Service**
2. **Consider privacy implications** of comment data
3. **Anonymize personal information** when possible
4. **Follow ethical research practices**
5. **Comply with local data protection laws** (GDPR, etc.)

## 🛠 Data Preprocessing

The framework automatically handles:
- Text cleaning and normalization
- Date parsing and validation
- Korean text preprocessing
- Quality filtering
- Duplicate removal

See `src/data_processor.py` and `src/data_filter.py` for details.

## 📞 Support

If you encounter issues with data formatting:
1. Check the sample files for reference
2. Review the data processor logs
3. Ensure your CSV encoding is UTF-8
4. Verify date formats are parseable

## 🔍 Data Quality Tips

For best analysis results:
- Include at least 1,000 comments
- Ensure comments span multiple time periods
- Include upvote/downvote data if available
- Use consistent date formatting
- Remove obvious spam/bot comments before analysis 