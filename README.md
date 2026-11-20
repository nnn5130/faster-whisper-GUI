# faster-whisper-GUI

    faster-whisper、whisperX，GUI with PySide6

- ## model download

  - https://huggingface.co/models?sort=trending&search=faster-whisper
  
  - you can also download and convert models in software

  - large-v3 model float32 :
  
    - [Huggingface](https://huggingface.co/CheshireCC/faster-whisper-large-v3-float32)
    
    - [百度云网盘链接](https://pan.baidu.com/s/1qltCehSq3pWMlIJ06sWLCQ?pwd=5xq8)
    
        
  
- ### Links

  - [pyside6-fluent-widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
  - [faster-whisper](https://github.com/guillaumekln/faster-whisper)
  - [whisperX](https://github.com/m-bain/whisperX)
  - [HuggingFace models download](https://huggingface.co/models)
  - [BandIt Plus](https://github.com/kwatcharasupat/bandit) — cinematic dialogue / music / effects separation
  - more and better separation tools:

    - [UVR](https://github.com/Anjok07/ultimatevocalremovergui#installation)
  
- ## What's this

  - this is a GUI software of faster-whisper , you can:
    - Transcrib audio or video files to srt/txt/smi/vtt/lrc file
    - provide all paraments of VAD-model and whisper-model
    - now, it support whisperX
    - BandIt Plus cinematic source separation (dialogue / music / effects)
    - whisper large-v3 model support

- ## Install

  Python 3.10 or 3.11. On Windows with an RTX 4070 (12GB), install the CUDA 12.1 PyTorch build. The old `torch==1.13.1+cu117` pin does not include Ada (sm_89) kernels, so that GPU will not run it.

  ```bash
  pip install -r requirements.txt
  ```

  `requirements.txt` pulls `torch==2.1.2+cu121` and `torchaudio==2.1.2+cu121` from the official PyTorch index, and keeps NumPy on 1.x because that PyTorch build does not support NumPy 2. You need an NVIDIA driver that supports CUDA 12.1 (driver 530 or newer). BandIt Plus uses CUDA when `torch.cuda.is_available()` is true and otherwise runs on CPU with the same install.

  A machine without an NVIDIA GPU can install the smaller CPU wheels instead of the lines in `requirements.txt`:

  ```bash
  pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
  ```

  Then install the other packages from `requirements.txt` without reinstalling torch. The CUDA wheel is the one the GUI is tested against, and it already falls back to CPU when no GPU is present.

  The first separation downloads the official BandIt Plus checkpoint (about 450MB) to `./cache/dnr-3s-mus64-l1snr-plus.ckpt`. That file is `dnr-3s-mus64-l1snr-plus.ckpt` from [Zenodo 10160601](https://zenodo.org/records/10160601). To use a copy you already have, put it at that path or set `BANDIT_PLUS_CHECKPOINT` to the file.

  Output stems are speech (dialogue), music, and effects. The default choice writes dialogue only, which replaces the old vocals-only option. "Dialogue and background" writes speech and everything else (music + effects). Segment length defaults to 6 seconds, the length the model was trained on. On a 12GB RTX 4070, 6–10 seconds is a comfortable range; longer chunks use more VRAM. Overlap is the fraction of each chunk that is crossfaded into the next (maximum 0.5).

---

- ## Best wishes to the world that received this message

  - ### Agreement

    - By using this software, you have read and agreed to the following user agreement:
      - You agree to use this software in compliance with the laws of your country or region.
      - You may not perform, including, but not limited to, the following acts, nor facilitate any violation of the law:
        - those who oppose the basic principles laid down in the Constitution.
        - endangering national security, divulging state secrets, subverting state power and undermining national unity.
        - harming the honor and interests of the country.
        - inciting ethnic hatred and racial discrimination.
        - those who sabotage the country's religious policy and promote cults.
        - spreading rumors, disturbing social order and undermining social stability.
        - spreading pornography, gambling, violence, murder, terrorism or abetting crime.
        - insulting or slandering others and infringing upon the legitimate rights and interests of others.
        - containing other contents prohibited by laws or administrative regulations.
      - All consequences and responsibilities caused by violations of laws and regulations in any related matters such as the generation, collection, processing and use of your data shall be borne by you.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=CheshireCC/faster-whisper-GUI&type=Timeline)](https://star-history.com/#CheshireCC/faster-whisper-GUI&Timeline)

- ### UI Language ###

    ![屏幕截图 2024-03-11 183130](./README.assets/183130.png)

- ### Theme Color ###

    ![屏幕截图 2024-03-11 184459](./README.assets/184459.png)

    ![image-20240311184818398](./README.assets/image-20240311184818398.png)

- ### Load Model / Download Model / Convert Model

![image-20231118155123131](./README.assets/image-20231118155123131.png)

- ### Large-v3 模型支持

  ![image-20231118155209847](./README.assets/image-20231118155209847.png)
- ### BandIt Plus cinematic separation

  Dialogue, music, and effects. The page still takes a file list, segment length, overlap, and an output directory. The screenshot below is the previous separation page; the stem list is now dialogue / music / effects.

  ![BandIt Plus separation page](./README.assets/DemucsFunction.png)

- ### batch process

![image-20231008150849827](./README.assets/image-20231008150849827.png)

- ### File List

  ![0.3.0_newFIleSystem](./README.assets/0.3.0_newFIleSystem.png)
- ### FileFilter

  ![fileFilter](./README.assets/fileFilter.png)
- ### WhisperX function

![0.3.0_whisperx](./README.assets/0.3.0_whisperx.png)

- ### paraments of faster-whisper model

![image-20231113020210745](./README.assets/image-20231113020210745.png)

- ### Silero VAD

  ![image-20231113020407272](./README.assets/image-20231113020407272.png)
- ### setting

  ![image-20231118155300816](./README.assets/image-20231118155300816.png)
- ### Show result and edit timestample

  ![0.3.0_result](./README.assets/0.3.0_result.png)![image-20231007191942864](./README.assets/image-20231007191942864.png)
- ### words-level timestamps —— karaoka lyric (work in `VTT`/`LRC`/`SMI` format)


  - play with foobar2000 , ESLyric plugin, `lrc` format lyric

  ![image-20230811130449688](./README.assets/image-20230811130449688.png)
