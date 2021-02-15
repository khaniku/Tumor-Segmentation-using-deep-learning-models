# Brain-Tumor-Segmentation

## Usage

### 1. Install Dependencies
```
pipenv install
pipenv shell
```

### 2. Set up .env
``` 
cp .env.example .env
```

### 3. Run Exp. with Command Line

```
python main.py -m <model_id> -d <data_provider_id> [--comet]
```

* Please refer to `models/__init__.py` for available model_ids,
and `data/data_providers.py` for available data_provider_ids.  
* For other arguments, please refer to `parser.py`
* Passing the `--comet` argument allows the user to log results to comet.ml, 
you'll have to add your api-key to the `.env` file

### 4. Resume Training from Checkpoint

```
python main.py --checkpoint_dir <checkpoint_dir>
``` 

## 5. Prediction

```
python predict.py --checkpoint_dir <checkpoint_dir> [--predict_mode] [--save_volume]
```
