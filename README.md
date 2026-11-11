# Detekcija feromagnetnih čestica

Domaći zadatak iz predmeta Programski jezici (OpenCV).

## Pokretanje

```
pip install -r requirements.txt
python detect.py images/ pred/
python grade.py gt/ pred/
```

Prvi argument je folder sa slikama, drugi je folder u koji se upisuju CSV fajlovi
(po jedan za svaku sliku, isto ime kao slika).

## Kako radi

1. **Mapa** - slika se prebaci u Lab prostor i saberu se svjetlina (L) i
   crvenkastost (a). Čestice su svijetle i bakarne, izolacija tamna.
2. **Maska** - Otsu prag (računa se posebno za svaku sliku), pa otvaranje i
   zatvaranje da se ukloni šum i popune rupe.
3. **Jezgra** - stroži prag, iz kojeg se black-hat filterom izrežu tamne
   pukotine između čestica. Tako svaka čestica dobije svoje jezgro.
4. **Watershed** - jezgra se šire nazad do ivice maske.
5. **Podjela** - regije koje nisu kompaktne (mala solidnost) vjerovatno sadrže
   više spojenih čestica. Preko distance transforma se nađe najuže mjesto i
   tu se regija razdvoji.
6. **Mjerenje** - broj piksela je površina, a `cv2.minAreaRect` daje centar,
   stranice i ugao orijentisanog pravougaonika.

Ne koristi se mašinsko učenje, a kod ne čita referentne (`gt`) fajlove.

## Rezultat

| slika | mean_f1 |
|-------|---------|
| V1 | 0.758 |
| V2 | 0.769 |
| V3 | 0.471 |
| V4 | 0.739 |
| V5 | 0.508 |
| V6 | 0.759 |
| **prosjek** | **0.667** |

Izlazni fajlovi su u folderu `pred/`.
