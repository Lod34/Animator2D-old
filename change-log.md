# Changelog - Animator2D

Tutte le modifiche significative al progetto *Animator2D* sono documentate in questo file.

## [Animator2D-v1.0.0] - In sviluppo
- **Sviluppo iniziato**: 6 marzo 2025 (post-v3.0.0-alpha)
- **Nuova direzione del progetto**:
  - Ripartenza da zero con un approccio modulare in tre fasi:
    1. **Creation**: Generazione o importazione di uno sprite base (esplorando modelli preesistenti o分解 in componenti come testa, braccia, ecc.).
    2. **Animation**: Configurazione dei parametri di animazione (es. azione, direzione, numero di frame) usando il dataset `pawkanarek/spraix_1024` o alternative.
    3. **Generation**: Produzione dell’output finale con scelta del formato (GIF, sprite sheet, video).
  - **Obiettivo**: Semplificare l’esperienza utente e migliorare la qualità separando la creazione dall’animazione.
  - **Stato**: In fase di ideazione e sperimentazione, non funzionante.

## [Animator2D-v3.0.0-alpha] - 2025-03-06
- **Sviluppo iniziato**: 6 marzo 2025
- **Rilascio**: 6 marzo 2025
- **Novità principali**:
  - Fix della versione *v2.0.0-alpha* con riscrittura parziale del codice.
  - Ottimizzazioni per il training del modello AI.
  - Aggiunta di *Residual Blocks* e *Self-Attention* al generatore per migliorare dettagli e coerenza.
  - *Frame Interpolator* ottimizzato per animazioni multi-frame (fino a 16 frame, 256x256 pixel).
  - Utilizzo di T5 come encoder testuale.
  - Interfaccia Gradio avanzata con controllo FPS e output GIF.
- **Miglioramenti tecnici**:
  - Addestramento con AdamW e scheduler Cosine Annealing su `pawkanarek/spraix_1024` (split 80/20 train/validation).
  - Risolto l’errore di importazione su Hugging Face Spaces presente in v2.0.0-alpha.
- **Stato**: Non funzionante.

## [Animator2D-v2.0.0-alpha] - 2025-03-03
- **Sviluppo iniziato**: 2 marzo 2025
- **Rilascio**: 3 marzo 2025
- **Novità principali**:
  - Riscrittura completa del codice rispetto a v1.
  - Nuovo approccio all’elaborazione delle animazioni.
  - Introduzione di T5 come encoder testuale e un *Frame Interpolator* per multi-frame.
  - Generatore più complesso rispetto a v1.
- **Problemi**:
  - Errore di importazione su Hugging Face (file `.pth` errato), output limitato a “pallina gialla su sfondo blu”.
- **Stato**: Non funzionante.

## [Animator2D-mini-v1.0.0-alpha] - 2025-03-01
- **Sviluppo iniziato**: 26 febbraio 2025
- **Rilascio**: 1 marzo 2025
- **Novità principali**:
  - Variante semplificata di v1 per test rapidi.
  - Utilizzo di CLIP come encoder e un generatore leggero.
  - Varianti:
    - *10e*: Allenamento su 10 epoche, risultati preliminari.
    - *100e*: Allenamento su 100 epoche, miglioramento visibile.
    - *250e*: Allenamento su 250 epoche, stabilità parziale.
- **Miglioramenti**:
  - Output a 64x64 o 128x128 pixel con progressi incrementali.
- **Stato**: Non funzionante.

## [Animator2D-v1.0.0-alpha] - 2025-02-22
- **Sviluppo iniziato**: 21 febbraio 2025
- **Rilascio**: 22 febbraio 2025
- **Novità principali**:
  - Prima versione sperimentale del modello AI.
  - Architettura base per animazioni 2D con BERT come encoder testuale e generatore semplice (64x64 pixel).
  - Prototipo iniziale con interfaccia Gradio e output simulato.
- **Stato**: Non funzionante.

---

**Note**: Le descrizioni della version _aplha_ riflettono i progressi tecnici, ma tutte le versioni restano non funzionanti in termini di output coerente per sprite animati.
