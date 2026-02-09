# Jupyter Notebook Parameter Preparation

## 📓 Two Notebooks Provided

### 1. `quick_start.ipynb` - ⚡ Fast & Simple
**Best for:** Quick parameter setup and immediate use

**Features:**
- Minimal documentation, maximum speed
- 4 template cells - run only the one you need
- Direct parameter editing
- Perfect for experienced users

**Workflow:**
1. Open notebook
2. Run setup cell
3. Edit ONE template cell with your values
4. Run that template cell
5. Run execution cell
6. Done!

---

### 2. `parameter_preparation.ipynb` - 📚 Complete & Detailed
**Best for:** Learning the system and reference

**Features:**
- Full documentation in each cell
- Detailed explanations of each parameter
- Preview and validation cells
- Parameter reference guide included
- Tips and examples throughout

**Workflow:**
1. Open notebook
2. Run setup cell
3. Navigate to your chosen template (1-4)
4. Customize parameters with inline guidance
5. Run template cell to configure
6. Select template in selection cell
7. Run execution cell
8. Optional: Run preview cells to inspect results

---

## 🎯 Quick Decision Guide

**Use `quick_start.ipynb` if you:**
- Know what you want to configure
- Want minimal reading/documentation
- Are familiar with the parameter structure
- Want to get running quickly

**Use `parameter_preparation.ipynb` if you:**
- Are new to the system
- Want to understand each parameter
- Need examples and tips
- Want detailed validation and preview

---

## 🚀 Getting Started

### Open in Jupyter

```bash
jupyter notebook quick_start.ipynb
# or
jupyter notebook parameter_preparation.ipynb
```

### Open in JupyterLab

```bash
jupyter lab
# Then navigate to the notebook in the file browser
```

### Open in VS Code

1. Open VS Code
2. Install Jupyter extension (if not installed)
3. Open the .ipynb file
4. Select Python kernel

---

## 📋 The 4 Templates Explained

Both notebooks contain the same 4 templates:

### Template 1: Single Year Monthly Mosaics
```python
ProdParams = {
    'year': 2023,
    'months': [6, 7, 8],
    'regions': './path/to/file.kml',
    # ... other params
}
```
**Creates:** Monthly windows for one year (e.g., June-Aug 2023)

---

### Template 2: Custom Date Ranges
```python
ProdParams = {
    'start_dates': ['2023-06-01', '2023-08-15'],
    'end_dates':   ['2023-06-30', '2023-09-15'],
    'regions': './path/to/file.kml',
    # ... other params
}
```
**Creates:** Exact date windows you specify

---

### Template 3: Single Date
```python
ProdParams = {
    'start_date': '2023-07-15',
    'regions': './path/to/file.kml',
    # ... other params
}
```
**Creates:** Single day window

---

### Template 4: Multi-Year Monthly
```python
ProdParams = {
    'year': 2019,
    'num_years': 5,
    'months': [6, 7, 8],
    'regions': './path/to/file.kml',
    # ... other params
}
```
**Creates:** Same months repeated across years (e.g., 5 years of summer)

---

## 🔧 Common Customizations

### Change Region File
```python
'regions': './your/path/to/regions.kml',  # or .shp
```

### Process Subset of Regions
```python
'regions_start_index': 0,      # Start at first region
'regions_end_index': 10,       # Stop at 10th region (None = all)
```

### Add Spatial Buffer
```python
'spatial_buffer_m': 100,       # 100 meter buffer around regions
```

### Add Temporal Buffer
```python
'temporal_buffer': [-5, 5],    # ±5 days around each date window
```

### Change Output Folder
```python
'out_folder': './output/my_analysis'
```

---

## 📊 Output Files

After running the preparation, you'll get:

1. **Validated Parameters** - Ready to pass to Production.py
2. **polygon_processing_log.csv** - In your output folder
   - Lists all polygons from your input file
   - Shows validity status (zero-area detection)
   - Contains area calculations

---

## 💡 Tips for Success

### 1. Test with Small Subsets First
```python
'regions_end_index': 5,  # Just process 5 regions to test
```

### 2. Use Temporal Buffers
Helps ensure image availability:
```python
'temporal_buffer': [-7, 7],  # ±7 days
```

### 3. Check File Paths
Use absolute paths if having issues:
```python
'regions': '/full/path/to/regions.kml',
```

### 4. Save Working Configurations
Copy successful ProdParams to a text file for reuse

### 5. Review the Log
Always check `polygon_processing_log.csv` before running full production

---

## 🔄 Typical Workflow

```
1. Open notebook
   ↓
2. Edit template parameters
   ↓
3. Run template cell
   ↓
4. Run execution cell
   ↓
5. Review validation output
   ↓
6. Check polygon_processing_log.csv
   ↓
7. Pass to Production.py:
   main(result['ProdParams'], result['CompParams'])
```

---

## ❓ Troubleshooting

### "FileNotFoundError: Region file not found"
- Check the path to your KML/SHP file
- Use absolute path instead of relative
- Verify file extension is correct

### "All polygons have zero area"
- Your KML/SHP file contains degenerate polygons
- Check the `polygon_processing_log.csv` for details
- Fix the source file or remove bad polygons

### "Invalid date format"
- Dates must be 'YYYY-MM-DD' format
- Example: '2023-06-15' not '06/15/2023'

### Template cell not working
- Make sure you ran the setup cell first
- Check that all required parameters are set
- Look for typos in parameter names

### Kernel died/crashed
- Check that all dependencies are installed
- Verify `prepare_params.py` is in the same directory
- Restart kernel and try again

---

## 🎨 Customization Examples

### Growing Season Analysis (5 Years)
```python
ProdParams = {
    'sensor': 'HLS_SR',
    'year': 2019,
    'num_years': 5,
    'months': [5, 6, 7, 8, 9],  # May through September
    'regions': './farms.kml',
    'spatial_buffer_m': 50,
    'temporal_buffer': [-5, 5],
    'out_folder': './output/growing_season'
}
```

### Pre/Post Event Comparison
```python
ProdParams = {
    'sensor': 'HLS_SR',
    'start_dates': ['2023-05-01', '2023-06-15'],  # Before and after
    'end_dates':   ['2023-05-15', '2023-06-30'],
    'regions': './event_area.kml',
    'spatial_buffer_m': 200,
    'out_folder': './output/event_comparison'
}
```

### High-Resolution Small Area
```python
ProdParams = {
    'sensor': 'HLS_SR',
    'year': 2023,
    'months': [6, 7, 8],
    'regions': './research_plot.kml',
    'regions_end_index': 1,  # Just first region
    'spatial_buffer_m': 20,
    'temporal_buffer': [-3, 3],
    'resolution': 10,  # Higher resolution
    'out_folder': './output/research'
}
```

---

## 📚 Next Steps

After successful parameter preparation:

```python
# In a new cell or script:
from Production import main

# Use the validated parameters
main(result['ProdParams'], result['CompParams'])
```

---

## 📞 Support

- Check the parameter reference in `parameter_preparation.ipynb`
- Review the `README.md` for detailed documentation
- See `QUICK_REFERENCE.txt` for command-line alternatives

---

## 🎯 Quick Reference

| Template | Use When | Key Parameters |
|----------|----------|----------------|
| 1 | Monthly analysis, one year | `year`, `months` |
| 2 | Custom time periods | `start_dates`, `end_dates` |
| 3 | Single date snapshot | `start_date` |
| 4 | Multi-year trends | `year`, `num_years`, `months` |

**All templates support:**
- `regions`: Path to KML/SHP file
- `regions_start_index` / `regions_end_index`: Region subset
- `spatial_buffer_m`: Buffer around regions
- `temporal_buffer`: Days before/after dates
- `out_folder`: Output directory

---

**Happy Processing! 🛰️**
