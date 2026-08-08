# Data Processing

**Goal:** convert VASP relaxation runs stored in Box/Dropbox into a clean set of INCAR/OUTCAR pairs for [`build-training-set.ipynb`](../mace-mh-1/build-training-set.ipynb) to convert into training data.

## Source datasets

- Dropbox: [O2DissociationSAA_2](https://www.dropbox.com/scl/fo/qpg1zuo3g7vb3il1wmqy3/AA4wERzz28lJhYTCAcSEvqk?rlkey=77cqlph0wcrovb1inqlccko0d&e=2&st=5zjrj8ry&dl=0)
- Box: [EO_Project_reactivity](https://tulane.box.com/s/osm9xd4zhr3pyq3g9m1oi7l1wjuqhn44), [EO_Project-reactivity_part2](https://tulane.box.com/s/pq2ciurncb2dvvoor0exz7t59ju0iag4)

## Usage

Run `ag-data-cleaning.ipynb` and then `ag-data-selection.ipynb`.

The final list of folder names for selected configs is cached as `screened_foldernames.txt`, which [`build-training-set.ipynb`](../mace-mh-1/build-training-set.ipynb) reads to determine which downloaded OUTCARs to use.

`screened_foldernames.txt` currently holds {Ag, C, H, O}-containing configs, `withNi_screened_foldernames.txt` holds {Ag, Ni, C, H, O}.

Auth cells are marked "one-time" in `ag-data-cleaning.ipynb`; run once per token refresh, not on every execution.

## Environment setup

Create a `.env` file in `data-processing/`:

```bash
# Dropbox credentials
DROPBOX_REFRESH_TOKEN=
APP_KEY=
APP_SECRET=
DROPBOX_TOKEN=

# Box credentials
BOX_CLIENT_ID=
BOX_CLIENT_SECRET=
DEV_TOKEN=
```

### Dropbox
Create a [developer account](https://www.dropbox.com/login?cont=https%3A%2F%2Fwww.dropbox.com%2Fdevelopers%2Fapps%3F_tk%3Dpilot_lp%26_ad%3Dtopbar4%26_camp%3Dmyapps), then create an app with the **Scoped** permission type. Under the **Permissions** tab, select:
- `files.content.read`
- `sharing.read`
- `file_requests.read`

Add your `App Key` and `App Secret` (found under the **Settings** tab) to `.env`. The remaining credentials are generated in `ag-data-cleaning.ipynb`.

### Box
Create a [developer account](https://app.box.com/developers/console), then create an app with the **App Access Only** permission type. Add your `Client ID` and `Client Secret` to `.env`, then generate a developer token and add that as well.

Note: Box has no persistent auth token — tokens expire after an hour and must be regenerated.