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
const port = process.env.PORT || "5000";

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(morgan("dev"));

app.use(
  fileUpload({
    createParentPath: true,
  })
);

app.use(express.static(path.join(__dirname, "client", "build")));

app.get("/*", (req, res) => {
  res.sendFile(path.join(__dirname, "client", "build", "index.html"));
});

let predictionResult = Object();

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
        if (!fs.existsSync("./upload")) {
          fs.mkdirSync("./upload");
          console.log("upload folder is created!");
        }
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
        // Start prediction
        let filename = audioFile.name;
        console.log(`Start prediction for ${filename}`);
        const pythonProcess = cp.spawn("python3", ["./predict.py", filename]);
        pythonProcess.stdout.on("data", (data) => {
          var dataStr = data.toString();
          dataStr = dataStr.replace(/\n/g, "").replace(/'/g, '"');
          predictionResult[filename] = dataStr;
        });
        pythonProcess.stderr.on("data", (data) => {
          console.error(`stderr: ${data}`);
        });
        pythonProcess.on("close", (code) => {
          console.log(`child process exited with code ${code}`);
          fs.unlink(`./upload/${filename}`, () => {
            console.log(`${filename} was deleted`);
          });
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

app.post("/check", (req, res) => {
  var filename = req.body.name;
  // var rootDir = path.join(__dirname, "../");
  try {
    let result = predictionResult[filename];
    if (result) {
      res.send({
        status: true,
        message: "Prediction results",
        code: 1,
        data: {
          name: filename,
          prediction: JSON.parse(result),
        },
      });
    } else {
      res.send({
        status: true,
        message: "No results",
        code: 0,
      });
    }
  } catch (err) {
    console.log(err);
    res.status(500).send(err);
  }
});

app.listen(port, () => {
  console.log(`listening at port: ${port}`);
});
