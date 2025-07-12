# yolo_dishes_project
yolo dishes finding

```
yolo_dishes_project/
├── data/
│   ├── raw/              # видео и кадры + сырые разметки
│   ├── processed/        # аугментированные изображения
│   └── dataset/          # итоговый комплект train/val/test
├── notebooks/            # при желании – для исследований
├── report/
│   ├── figures/          # графики и картинки для отчёта
│   └── report.md         # итоговый отчёт
├── runs/                 # создаётся Ultralytics-ом
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── extract_frames.py
│   │   ├── augment.py
│   │   └── split_dataset.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── dataset.yaml   # конфиг для YOLO
│   ├── utils/
│   │   ├── logger.py
│   │   └── metrics.py
│   ├── report/
│   │   └── make_report.py
│   └── main.py            # orchestration CLI
├── requirements.txt
└── README.md
```

1.	<b>Скопировать видео</b>	Вы скачиваете файл data/raw/video.mp4	video.mp4
2.  <b>Извлечь кадры</b>	<code>python -m src.data.extract_frames	data/raw/frames/*.jpg</code>
3.	<b>Аннотировать</b>	открыть labelImg → YOLO-txt	data/raw/labels/*.txt
4.	<b>Аугментация</b>	<code>ppython -m src.data.augment</code>	data/processed/*.jpg + .txt
5.	<b>Сплит</b>	<code>python -m src.data.split_dataset</code>	data/dataset/{train,val,test}/images,labels
6.	<b>Тренировка</b>	<code>python -m src.models.train</code>	runs/yolo11_exp*/weights/best.pt, ( также с флагами:

<code>python -m src.models.train \
  --size s \
  --epochs 50 \
  --img 960 \
  --batch 8 \
  --lr0 0.005 </code>
 
<code>python -m src.models.train \
       --size x \
       --epochs 50 \
       --batch 32 \
       --lr0 0.005</code>)
7.	<b>Оценка</b>	<code>python -m src.models.evaluate</code> (берет самую последнюю best.pt), либо <code>python -m src.models.evaluate \
       --weights runs/exp11_s/weights/best.pt</code>	метрики metrics.json
8.	<b>Tuning ×2</b>	правим config.py, повторяем 7-8	3 эксперимента, сравнительная таблица
9.	<b>Отчёт</b>	<code>python -m src.report.make_report</code>	report/report.md, графики report/figures


## YOLOv11 Dish Detection Pipeline

### Как запустить

```bash
# 1. Создать виртуальное окружение
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Запустить полный пайплайн
python -m src.main all --video /absolute/path/to/video.mp4
```

После выполнения в каталоге `report/` появится готовый `report.md` + графики.
