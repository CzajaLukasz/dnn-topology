import os
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Inicjalizacja pobierania z Hugging Face (flwrlabs/pacs)...")
    # Ładujemy zbiór z Hugging Face
    dataset = load_dataset("flwrlabs/pacs")
    
    # Mapowanie nazw domen w tym zbiorze (zazwyczaj splits lub kolumna 'domain')
    # Dla flwrlabs/pacs sprawdzimy dostępne podzbiory
    base_dir = "./data/pacs"
    os.makedirs(base_dir, exist_ok=True)

    # flwrlabs/pacs ma podział na train/validation/test, ale zawiera kolumnę 'domain' oraz 'label'
    # Przechodzimy po wszystkich splitach (train, validation, test), żeby wyciągnąć 100% danych
    splits = dataset.keys()
    
    # Słownik do mapowania id klas na nazwy tekstowe (PACS ma 7 klas)
    categories = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    print("Rozpakowywanie i układanie obrazów w strukturze ImageFolder...")
    for split in splits:
        print(f"Przetwarzanie sekcji: {split}...")
        # Dodajemy enumerate, aby mieć bezpieczny, unikalny licznik (idx) dla każdego obrazu
        for idx, item in tqdm(enumerate(dataset[split]), total=len(dataset[split])):
            # Wyciągamy informacje o domenie, klasie oraz surowy obiekt PIL Image
            domain = item['domain']  # np. 'art_painting', 'cartoon'
            label_id = item['label'] # indeks 0-6
            image = item['image']    # Obiekt PIL.Image
            
            class_name = categories[label_id]
            
            # Tworzymy ścieżkę: ./datasets/pacs/nazwa_domeny/nazwa_klasy/
            target_dir = os.path.join(base_dir, domain, class_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # Bezpieczna nazwa pliku bazująca na splicie i indeksie pętli
            filename = f"{split}_{idx}.jpg"
            filepath = os.path.join(target_dir, filename)
            
            # Zapisujemy jako JPG (konwertując do RGB na wypadek nietypowych formatów)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(filepath, "JPEG")

    print("\nSukces! Zbiór PACS został bezpiecznie pobrany i ułożony w: ./datasets/pacs/")

if __name__ == "__main__":
    main()