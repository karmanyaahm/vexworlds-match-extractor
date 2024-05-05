


make sure you have these dependencies:
```
          command line - ffmpeg
          python - opencv-python
          python - pytesseract
          python - yt-dlp

```

for example to extract match 185 of the vexu innovate division, give it the links to the three streams, the match number, run this command in an empty folder, and it'll cook
```
python extract.py -i https://www.vexworlds.tv/#/viewer/broadcasts/qualification-matches-innovate-yxlmbfpfa3oj275czjmf/ulkuufu5cam4s06gdim3 -i https://www.vexworlds.tv/#/viewer/broadcasts/qualification-matches-innovate-elk09hemvm4c9ewwwevn/o0pngqswkl9vdmx59rwp -i https://www.vexworlds.tv/#/viewer/broadcasts/qualification-matches-innovate-pqsqpx5qx9ejnwpgk6zn/ngnvh73ovzgrtrnut9er -m 185
```

only tested on my very specific linux install, but should probably work on other unixy things with zero to tiny modifications

warning that it downloads the whole stream before cutting so it'll take up like 30GB of space
