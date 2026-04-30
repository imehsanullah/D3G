# Downloading D3G / MetaGraspNetV2 Datasets

This documents the dataset setup used for this D3G checkout.

## Key Paths

The repo expects dataset files under `./datasets` by default:

```yaml
DATASETS:
  ROOT: './datasets'
```

The repo partition had limited free space, so the actual data was placed on `/tmp` and symlinked back into the repo:

```bash
cd /home/user/ehsanullahm1/thesis/D3G
mkdir -p /tmp/d3g_datasets /tmp/d3g_downloads
ln -s /tmp/d3g_datasets datasets
```

Result:

```text
D3G/datasets -> /tmp/d3g_datasets
```

Downloads are staged in:

```text
/tmp/d3g_downloads
```

Extracted data lives in:

```text
/tmp/d3g_datasets
```

## Author-Provided D3G Metadata

D3G-specific splits and metadata are stored in Google Drive:

```text
https://drive.google.com/drive/folders/1e9_Oa05Cdt5K4aa3rRRf__t5l5ozUeZf
```

Install `gdown` if needed:

```bash
conda run -n d3g python -m pip install gdown
```

Download metadata into `D3G/data`:

```bash
cd /home/user/ehsanullahm1/thesis/D3G
conda run -n d3g gdown --folder \
  https://drive.google.com/drive/folders/1e9_Oa05Cdt5K4aa3rRRf__t5l5ozUeZf \
  -O data
```

Expected files:

```text
data/sample_metadata.json
data/sample_real_metadata.json
data/scene_real_metadata.json
data/scene_synt_metadata.json
data/splits.json
```

Observed metadata sizes:

```text
249M data/sample_metadata.json
508K data/sample_real_metadata.json
28K  data/scene_real_metadata.json
328K data/scene_synt_metadata.json
32K  data/splits.json
```

## MetaGraspNetV2 Source Links

The MetaGraspNetV2 README points to these public Nextcloud shares:

```text
MGN-Sim:  https://nx25922.your-storageshare.de/s/pDik4DYzps2grKC
MGN-Real: https://nx25922.your-storageshare.de/s/HPe2eYGGJGz6rqd
```

Direct WebDAV roots:

```text
MGN-Sim:  https://nx25922.your-storageshare.de/public.php/dav/files/pDik4DYzps2grKC/
MGN-Real: https://nx25922.your-storageshare.de/public.php/dav/files/HPe2eYGGJGz6rqd/
```

Archive totals from WebDAV listing:

```text
MGN-Sim:  81 zip chunks, about 820 GB compressed
MGN-Real: 17 zip chunks, about 27.7 GB compressed
```

## Query WebDAV Listings

Fetch folder listings:

```bash
curl -s -X PROPFIND -H 'Depth: 1' \
  https://nx25922.your-storageshare.de/public.php/dav/files/pDik4DYzps2grKC/ \
  -o /tmp/d3g_downloads/synth_propfind.xml

curl -s -X PROPFIND -H 'Depth: 1' \
  https://nx25922.your-storageshare.de/public.php/dav/files/HPe2eYGGJGz6rqd/ \
  -o /tmp/d3g_downloads/real_propfind.xml
```

Generate URL lists:

```bash
python - <<'PY'
import xml.etree.ElementTree as ET
from urllib.parse import quote

ns = {'d': 'DAV:'}

for dataset, token, xml_path in [
    ('real', 'HPe2eYGGJGz6rqd', '/tmp/d3g_downloads/real_propfind.xml'),
    ('synth', 'pDik4DYzps2grKC', '/tmp/d3g_downloads/synth_propfind.xml'),
]:
    root = ET.parse(xml_path).getroot()
    names = []
    for resp in root.findall('d:response', ns):
        href = resp.find('d:href', ns).text
        length = resp.find('.//d:getcontentlength', ns)
        if length is not None:
            name = href.rsplit('/', 1)[-1]
            if name.endswith('.zip'):
                names.append(name)

    def key(name):
        return int(name.removeprefix('data_ifl_').removesuffix('.zip'))

    names = sorted(names, key=key)
    out = f'/tmp/d3g_downloads/{dataset}_urls.txt'
    with open(out, 'w') as f:
        for name in names:
            f.write(f'https://nx25922.your-storageshare.de/public.php/dav/files/{token}/{quote(name)}\n')
    print(dataset, len(names), out, names[:3], names[-3:])
PY
```

Expected output:

```text
real 17 /tmp/d3g_downloads/real_urls.txt ['data_ifl_0.zip', 'data_ifl_1.zip', 'data_ifl_2.zip'] ['data_ifl_14.zip', 'data_ifl_15.zip', 'data_ifl_16.zip']
synth 81 /tmp/d3g_downloads/synth_urls.txt ['data_ifl_0.zip', 'data_ifl_1.zip', 'data_ifl_2.zip'] ['data_ifl_78.zip', 'data_ifl_79.zip', 'data_ifl_80.zip']
```

## Directory Layout Required by D3G Metadata

The D3G metadata resolves paths relative to `cfg.DATASETS.ROOT`, for example:

```text
MetaGraspNetV2_Synth/media/flairop/data11/data_ifl/scene0/0_rgb.png
MetaGraspNetV2_Real/mnt/data1/data_ifl_real/scene352/3_rgb.png
```

Create target directories:

```bash
mkdir -p \
  /tmp/d3g_datasets/MetaGraspNetV2_Synth/media/flairop/data11/data_ifl \
  /tmp/d3g_datasets/MetaGraspNetV2_Real/mnt/data1/data_ifl_real
```

## Minimal Synthetic Chunk

An earlier local dummy-test commit temporarily added this hardcoded filter:

```python
sample = sample[sample['scene'] < 100]
```

That filter has now been removed from `data/metagraspnet_synth_mapper.py`. If that temporary filter is present in another checkout, `data_ifl_0.zip` is enough to load that limited synthetic subset because it contains scenes `0..99`.

Download the first synthetic chunk:

```bash
mkdir -p /tmp/d3g_downloads/synth
wget -c --tries=0 --timeout=30 --waitretry=10 \
  -O /tmp/d3g_downloads/synth/data_ifl_0.zip \
  https://nx25922.your-storageshare.de/public.php/dav/files/pDik4DYzps2grKC/data_ifl_0.zip
```

Inspect:

```bash
unzip -l /tmp/d3g_downloads/synth/data_ifl_0.zip | sed -n '1,80p'
```

The synthetic ZIP contains `scene*/...` directly, so extract into:

```bash
unzip -q -o /tmp/d3g_downloads/synth/data_ifl_0.zip \
  -d /tmp/d3g_datasets/MetaGraspNetV2_Synth/media/flairop/data11/data_ifl
```

## Real Dataset Download and Extraction

Download all real chunks:

```bash
mkdir -p /tmp/d3g_downloads/real
wget -c -nv --tries=0 --timeout=30 --waitretry=10 \
  -P /tmp/d3g_downloads/real \
  -i /tmp/d3g_downloads/real_urls.txt
```

Extract real chunks:

```bash
for z in /tmp/d3g_downloads/real/data_ifl_*.zip; do
  unzip -q -o "$z" \
    -d /tmp/d3g_datasets/MetaGraspNetV2_Real/mnt/data1/data_ifl_real
done
```

The real ZIPs contain an extra internal prefix:

```text
mnt/data1/data_ifl_real/scene*/
```

After extraction, move the scene folders up one level:

```bash
find /tmp/d3g_datasets/MetaGraspNetV2_Real/mnt/data1/data_ifl_real/mnt/data1/data_ifl_real \
  -mindepth 1 -maxdepth 1 \
  -exec mv -t /tmp/d3g_datasets/MetaGraspNetV2_Real/mnt/data1/data_ifl_real {} +

rmdir \
  /tmp/d3g_datasets/MetaGraspNetV2_Real/mnt/data1/data_ifl_real/mnt/data1/data_ifl_real \
  /tmp/d3g_datasets/MetaGraspNetV2_Real/mnt/data1/data_ifl_real/mnt/data1 \
  /tmp/d3g_datasets/MetaGraspNetV2_Real/mnt/data1/data_ifl_real/mnt
```

## Full Synthetic Dataset Job

Full synthetic training beyond `scene < 100` requires all 81 synthetic chunks.

The script used for a resumable full synthetic download/extract job:

```bash
cat > /tmp/d3g_downloads/full_synth_job.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

LOG=/tmp/d3g_downloads/full_synth_job.log
TARGET=/tmp/d3g_datasets/MetaGraspNetV2_Synth/media/flairop/data11/data_ifl
DOWNLOAD_DIR=/tmp/d3g_downloads/synth
URLS=/tmp/d3g_downloads/synth_urls.txt

mkdir -p "$TARGET" "$DOWNLOAD_DIR"

echo "[$(date -Is)] starting synthetic download" >> "$LOG"
wget -c -nv --tries=0 --timeout=30 --waitretry=10 -P "$DOWNLOAD_DIR" -i "$URLS" >> "$LOG" 2>&1

echo "[$(date -Is)] download complete, starting extraction" >> "$LOG"
for z in "$DOWNLOAD_DIR"/data_ifl_*.zip; do
  echo "[$(date -Is)] extracting $(basename "$z")" >> "$LOG"
  unzip -q -o "$z" -d "$TARGET"
done

echo "[$(date -Is)] synthetic dataset job complete" >> "$LOG"
SH

chmod +x /tmp/d3g_downloads/full_synth_job.sh
```

Start it in tmux:

```bash
tmux new-session -d -s d3g_synth_download /tmp/d3g_downloads/full_synth_job.sh
```

Monitor progress:

```bash
tmux ls
tmux attach -t d3g_synth_download
tail -f /tmp/d3g_downloads/full_synth_job.log
du -sh /tmp/d3g_downloads/synth /tmp/d3g_datasets/MetaGraspNetV2_Synth
df -h /tmp
```

Completed run status:

```text
tmux session: no longer running
/tmp/d3g_downloads/synth: 81 zip chunks downloaded
/tmp/d3g_downloads/real: 17 zip chunks downloaded
/tmp/d3g_datasets/MetaGraspNetV2_Synth: extracted, 8008 scene directories
/tmp/d3g_datasets/MetaGraspNetV2_Real: extracted, 818 scene directories
```

## Verification Commands

Check disk usage:

```bash
du -sh /tmp/d3g_downloads/synth /tmp/d3g_downloads/real \
       /tmp/d3g_datasets/MetaGraspNetV2_Synth /tmp/d3g_datasets/MetaGraspNetV2_Real
df -h /tmp
```

Verify synthetic mapper:

```bash
cd /home/user/ehsanullahm1/thesis/D3G
conda run -n d3g python -c "import data; ds=data.metagraspnet_synth_mapper.get_metagraspnet_dict_synth('train'); print('train records', len(ds)); first=next(x for x in ds if x['scene'] < 100); print(first['scene'], first['rgb_path']); mapper=data.metagraspnet_synth_mapper.MetaGraspNetV2Mapper(data_root='./datasets', is_train=False, graph_gt_type='dense'); out=mapper(first); print(out['image'].shape, out['instances'].gt_boxes.tensor.shape, out['dense_gt'].shape)"
```

Observed result:

```text
train records 2960
78 MetaGraspNetV2_Synth/media/flairop/data11/data_ifl/scene78/0_rgb.png
torch.Size([3, 512, 512]) torch.Size([4, 4]) torch.Size([4, 4])
```

Verify real mapper:

```bash
cd /home/user/ehsanullahm1/thesis/D3G
conda run -n d3g python -c "import data; real=data.metagraspnet_real_mapper.get_metagraspnet_dict_real('test_all'); print('real records', len(real)); mapper=data.metagraspnet_real_mapper.MetaGraspNetV2MapperReal(data_root='./datasets', is_train=False, graph_gt_type='dense'); out=mapper(real[0]); print(real[0]['scene'], real[0]['rgb_path']); print(out['image'].shape, out['instances'].gt_boxes.tensor.shape, out['dense_gt'].shape)"
```

Observed result:

```text
real records 690
352 MetaGraspNetV2_Real/mnt/data1/data_ifl_real/scene352/3_rgb.png
torch.Size([3, 512, 512]) torch.Size([6, 4]) torch.Size([6, 6])
```

## Training Notes

The README does not appear to provide final trained D3G relationship-reasoning weights. It provides:

```text
public COCO checkpoints
fixed-key pretrain checkpoints
```

So final relationship-reasoning training should be run locally after dataset setup.

After removing the temporary `scene < 100` filter, the full downloaded synthetic splits verify as:

```text
synthetic train: 176006 records
synthetic val: 9287 records
synthetic test_easy: 18648 records
synthetic test_medium: 1480 records
synthetic test_hard: 444 records
```
