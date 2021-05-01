const express = require("express");
const fileUpload = require("express-fileupload");
const cors = require("cors");
const bodyParser = require("body-parser");
const morgan = require("morgan");
const _ = require("lodash");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");

const app = express();
const port = 3001;

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(morgan("dev"));

app.use(
  fileUpload({
    createParentPath: true,
  })
);

app.post("/upload", (req, res) => {
  try {
    if (!req.files) {
      res.send({
        status: false,
        message: "No file uploaded",
      });
    } else {
      var audioFile = req.files.audio;
      try {
        fs.writeFileSync(
          `upload/${audioFile.name}`,
          Buffer.from(new Uint8Array(audioFile.data))
        ); // write the blob to the server as a file
        res.send({
          status: true,
          message: "File is uploaded",
          data: {
            name: audioFile.name,
            mimetype: audioFile.mimetype,
            size: audioFile.size,
          },
        });
      } catch (err) {
        console.log(err);
        res.status(500).send(err);
      }
    }
  } catch (err) {
    console.log(err);
    res.status(500).send(err);
  }
});

app.post("/predict", (req, res) => {
  try {
    var filename = req.body.name;
    var rootDir = path.join(__dirname, "../");
    try {
      const pythonProcess = cp.spawn(
        "/home/xmliszt/Documents/git/dl-big-project/venv/bin/python",
        [path.join(rootDir, "predict.py"), filename]
      );
      pythonProcess.stdout.on("data", (data) => {
        var dataStr = data.toString();
        dataStr = dataStr.replace(/\n/g, "").replace(/'/g, '"');
        res.send({
          status: true,
          message: "Prediction results",
          data: {
            name: filename,
            prediction: JSON.parse(dataStr),
          },
        });
      });
      pythonProcess.stderr.on("data", (data) => {
        console.error(`stderr: ${data}`);
        res.status(500).send(err);
        return;
      });
      pythonProcess.on("close", (code) => {
        console.log(`child process exited with code ${code}`);
      });
    } catch (err) {
      console.log(err);
      res.status(500).send(err);
    }
  } catch (err) {
    console.log(err);
    res.status(500).send(err);
  }
});

app.listen(port, () => {
  console.log(`listening at http://localhost:${port}`);
});