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
    // var rootDir = path.join(__dirname, "../");
    try {
      const pythonProcess = cp.spawn("python3", ["./predict.py", filename]);
      pythonProcess.stdout.on("data", (data) => {
        var dataStr = data.toString();
        dataStr = dataStr.replace(/\n/g, "").replace(/'/g, '"');
        fs.unlink(`./upload/${filename}`, (err) => {
          if (err) throw err;
          console.log(`${filename} was deleted`);
        });
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
  console.log(`listening at port: ${port}`);
});
