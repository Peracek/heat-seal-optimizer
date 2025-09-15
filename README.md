# Optimalizátor parametrů tepelného svařování

Webová aplikace pro optimalizaci parametrů tepelného svařování Doypack obalů s využitím strojového učení.

## Funkce

- 🎯 **Optimalizace parametrů**: Nalezení optimálních nastavení teploty, tlaku a doby zdržení
- 📊 **Správa dat**: Kombinace historických CSV dat s ručním vstupem nových záznamů
- 🤖 **Strojové učení**: Random Forest model pro predikci úspěšnosti svařování
- 🇨🇿 **České rozhraní**: Plně lokalizované uživatelské rozhraní
- 💾 **Perzistentní úložiště**: SQLite databáze pro uložení uživatelských dat

## Struktura aplikace

### Hlavní stránka - Optimalizace parametrů
- Vstupní parametry (typ materiálu, typ barvy, pokrytí tiskem)
- Doporučené parametry svařování
- Předpokládaná úspěšnost

### Správa dat
- Přidávání nových produkčních dat
- Zobrazení a analýza existujících záznamů
- Ovládání přetrénování modelu

## Instalace a spuštění

### Požadavky
- Python 3.8+
- pip

### Kroky pro spuštění

1. **Klonování repozitáře**
   ```bash
   git clone <repository-url>
   cd doypack-streamlit
   ```

2. **Vytvoření virtuálního prostředí**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Na Windows: venv\\Scripts\\activate
   ```

3. **Instalace závislostí**
   ```bash
   pip install -r requirements.txt
   ```

4. **Spuštění aplikace**
   ```bash
   streamlit run app.py
   ```

5. **Otevření v prohlížeči**
   - Aplikace se automaticky otevře na `http://localhost:8501`
   - Pokud se neotevře automaticky, zkopírujte URL z terminálu

## Struktura souborů

```
doypack-streamlit/
├── app.py                 # Hlavní Streamlit aplikace
├── historical_data.csv    # Historická produkční data
├── requirements.txt       # Python závislosti
├── user_data.db          # SQLite databáze (vytváří se automaticky)
├── seal_model.pkl        # Trénovaný model (vytváří se automaticky)
├── encoder.pkl           # One-hot encoder (vytváří se automaticky)
└── README.md             # Tento soubor
```

## Použití

### První spuštění
1. Aplikace automaticky natrénuje model z `historical_data.csv`
2. Přejděte na stránku "Optimalizace parametrů"
3. Nastavte vstupní parametry a klikněte na "Najít optimální nastavení"

### Přidání vlastních dat
1. Přejděte na stránku "Správa dat"
2. V záložce "Přidat nová data" vyplňte formulář
3. Data se automaticky uloží do databáze
4. Model se přetrénuje při dalším použití

### Zobrazení dat
1. V záložce "Zobrazit data" můžete:
   - Prohlížet všechna data (CSV + manuální)
   - Stahovat kompletní dataset
   - Přetrénovat model manuálně
   - Zobrazit statistiky datasetu

## Technické detaily

### Strojové učení
- **Algoritmus**: Random Forest Classifier
- **Vstupy**: Typ materiálu, typ barvy, pokrytí tiskem, teplota, tlak, doba zdržení
- **Výstup**: Pravděpodobnost úspěšného svařování
- **Optimalizace**: Grid search pro nalezení optimálních parametrů

### Data
- **CSV data**: Historické produkční záznamy
- **Manuální data**: Uživatelsky přidané záznamy v SQLite databázi
- **Kombinace**: Model trénuje na sloučených datech z obou zdrojů

### Rozhraní
- **Framework**: Streamlit
- **Jazyk**: Čeština s anglickými daty pro kompatibilitu
- **Responzivní**: Optimalizováno pro desktop i mobilní zařízení

## Řešení problémů

### Model se nenatrenuje
- Zkontrolujte, zda existuje soubor `historical_data.csv`
- Přidejte alespoň několik záznamů přes "Správa dat"

### Aplikace se nespustí
- Ověřte, že máte aktivované virtuální prostředí
- Zkontrolujte instalaci závislostí: `pip install -r requirements.txt`

### Chyby v datech
- Zkontrolujte formát CSV souboru (viz `historical_data.csv` jako vzor)
- Ověřte rozsahy parametrů při ručním zadávání

## Kontribuce

Pro vývoj a úpravy aplikace:

1. Aktivujte virtuální prostředí
2. Proveďte změny v `app.py`
3. Testujte lokálně pomocí `streamlit run app.py`
4. Commitujte změny s popisnými commit zprávami

## Licence

MIT License - viz LICENSE soubor pro více detailů.
