# Root_Cause_Analysis_Langgraph
A beginner-friendly tool for systematic problem analysis using AI-powered Fishbone Diagrams (also known as Ishikawa Diagrams).

## What is a Fishbone Diagram?

Think of it as a detective tool that helps you find the **real** reasons behind problems. It looks like a fish skeleton, where:
- The **head** is your problem (the effect)
- The **bones** are different categories of potential causes
- The **smaller bones** are the specific causes and their root reasons

## How It Works

The tool uses the **6M Framework** to categorize potential causes:

1. **Man (People)** - Human factors, skills, training issues
2. **Machine** - Equipment, tools, technology failures  
3. **Method** - Process, procedures, workflow problems
4. **Material** - Raw materials, supplies, input quality
5. **Measurement** - Metrics, KPIs, monitoring gaps
6. **Environment** - External factors, conditions, culture

For each cause found, it asks "Why?" multiple times to dig deeper and find the true root causes.

## Example Output

```
Effect: "Mobile app release delays"
├── Machine: Server crashes
│   └── Why? Insufficient memory, No load balancing
├── Method: Poor testing process
│   └── Why? Rushed timelines, No test automation
├── Man (People): Inexperienced developers
│   └── Why? High turnover, Limited training budget
```

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key
Create a `.env` file in the project folder:
```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3. Run the Tool
```bash
python fishbone.py
```

### 4. Use It!
- Enter your problem when prompted
- Wait for AI analysis (usually 10-30 seconds)
- View the results on screen
- Find your analysis saved as a JSON file

## What You Get

- **Interactive Analysis**: Just type your problem and get instant results
- **Deep Root Cause Analysis**: Goes beyond surface-level causes
- **Structured Output**: Results saved as JSON files with timestamps
- **Visual Display**: Easy-to-read tree structure in your terminal

## Example Problems to Try

- "Website is loading slowly"
- "Customer complaints are increasing"
- "Team missing project deadlines"
- "High employee turnover"
- "Product quality issues"

## Files Created

Each analysis creates a timestamped JSON file like:
- `fishbone_analysis_20250915_163449.json`

These files contain all the causes and root causes found, perfect for sharing with your team or including in reports.

## Tips for Best Results

1. **Be Specific**: Instead of "problems with app", try "users can't login to mobile app"
2. **Focus on Effects**: Describe what's happening, not what you think causes it
3. **One Problem at a Time**: Analyze each issue separately for clearer results

## Troubleshooting

**"Error: Please set OPENAI_API_KEY"**
- Make sure your `.env` file exists and contains your API key

**"Analysis takes too long"**
- Complex problems may take 30-60 seconds to analyze
- Check your internet connection

**"No causes found"**
- Try rephrasing your problem more specifically
- Ensure the problem description is clear and detailed

## What's Next?

Once you have your fishbone analysis:
1. Review the root causes with your team
2. Prioritize which causes to address first
3. Create action plans for the most critical root causes
4. Use the JSON files for documentation and tracking

---
