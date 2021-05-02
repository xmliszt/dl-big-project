import { Upload, message, Button, Spin, Modal, Space, Progress } from "antd";
import { InboxOutlined, GithubOutlined } from "@ant-design/icons";
import "./Home.css";
import { useEffect, useState } from "react";
import axios from "axios";

const env = process.env.NODE_ENV;

const baseUrl =
  env === "production"
    ? "https://genrewiz.herokuapp.com"
    : "http://localhost:5000";

console.log("Base URL is: " + baseUrl);

export default function Home() {
  const [viewResults, setViewResults] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [uploadedFilename, setUploadedFilename] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [fileList, setFileList] = useState([]);
  const [startCheck, setStartCheck] = useState(false);
  const [secondsElapsed, setSecondsElapsed] = useState(0);

  useEffect(() => {
    if (secondsElapsed % 30 === 0 && secondsElapsed !== 0) {
      message.error(
        "Heroku server has timeout. Please refresh the page and retry."
      );
    }
  }, [secondsElapsed]);

  useEffect(() => {
    let checker;
    if (startCheck === true) {
      checker = setInterval(() => {
        checkResults(uploadedFilename);
        setSecondsElapsed((secondsElapsed) => secondsElapsed + 1);
      }, 1000);
    }
    return () => clearInterval(checker);
  }, [startCheck]);

  const props = {
    name: "audio",
    multiple: false,
    maxCount: 1,
    action: baseUrl + "/upload",
    onChange(info) {
      const { status } = info.file;
      let list = [...info.fileList];
      list = list.map((file) => {
        if (file.response) {
          file.url = file.response.url;
        }
        return file;
      });
      var fileType = info.file.name.split(".").pop().toLowerCase();
      if (fileType !== "wav") {
        message.error("Only WAV file is allowed");
        setFileList = [];
        return;
      }
      if (status === "done") {
        setViewResults(false);
        message.success(`${info.file.name} file uploaded successfully.`);
        setUploadedFilename(info.file.name);
        setLoading(true);
        setStartCheck(true);
      } else if (status === "error") {
        setViewResults(false);
        message.error(`${info.file.name} file upload failed.`);
      } else if (status === "removed") {
        message.warning(`${info.file.name} file removed successfully.`);
        setViewResults(false);
      }
      setFileList(list);
    },
    progress: {
      strokeColor: {
        "0%": "#108ee9",
        "100%": "#87d068",
      },
      strokeWidth: 3,
      format: (percent) => `${parseFloat(percent.toFixed(2))}%`,
    },
  };

  const checkResults = async (filename) => {
    try {
      let response = await axios.post(baseUrl + "/check", {
        name: filename,
      });
      var code = response.data.code;
      console.log("Code: " + code);
      if (code === 1) {
        setLoading(false);
        var predictions = response.data.data.prediction;
        setPredictions(predictions);
        setViewResults(true);
        setStartCheck(false);
      } else {
        setLoading(true);
      }
    } catch (err) {
      setLoading(false);
      console.error(err);
      message.error(err);
      setStartCheck(false);
    }
  };

  const onViewResults = () => {
    setShowResult(true);
  };

  const handleCloseResult = () => {
    setShowResult(false);
  };

  var ViewResults;
  var Loading;
  if (viewResults) {
    ViewResults = (
      <div className="upload-button-wrapper">
        <Button type="primary" size="large" onClick={onViewResults}>
          View Results
        </Button>
      </div>
    );
  }
  if (loading) {
    Loading = (
      <div style={{ marginTop: "1rem" }}>
        <Space>
          <Spin spinning={loading} size="default" />
          <span> AI is predicting the genres for you...</span>
        </Space>
      </div>
    );
  }
  return (
    <div className="home-wrapper">
      <Modal
        title="Our AI thinks that your song's genre is..."
        visible={showResult}
        onCancel={handleCloseResult}
        onOk={handleCloseResult}
        footer={
          <Button type="primary" onClick={handleCloseResult}>
            OK
          </Button>
        }
      >
        {predictions.map((prediction) => (
          <div className="result-wrapper">
            <div className="genre-text">{prediction.genre}</div>
            <div className="progress-wrapper">
              <Progress
                strokeColor={{
                  "0%": "#108ee9",
                  "100%": "#87d068",
                }}
                percent={(prediction.score * 100).toFixed(2)}
              />
            </div>
          </div>
        ))}
      </Modal>

      <div>
        <div className="title-style">
          <span>GenreWiz: Your Genre Classifier</span>
        </div>
        <div className="subtitle-style">
          <span>Upload your music now!</span>
        </div>
        <div className="upload-container">
          <div className="upload-wrapper">
            <Upload {...props} fileList={fileList}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined height="1rem" width="1rem" />
              </p>
              <p className="ant-upload-text">
                Click or drag your audio file to this area to upload
              </p>
              <p className="ant-upload-hint">
                Only support .wav (WAV) audio file type. Single file upload.
              </p>
            </Upload>
          </div>
        </div>
        {Loading}
        {ViewResults}
      </div>
      <div className="footer-wrapper">
        <Space>
          <div
            style={{
              fontSize: "1.5rem",
              color: "rgb(71, 71, 71)",
              opacity: 0.7,
            }}
          >
            <GithubOutlined
              style={{
                "::hover": {
                  cursor: "pointer",
                  opacity: 1,
                  transition: "opacity 0.3s",
                },
              }}
              onClick={() => {
                window.open("https://github.com/xmliszt/dl-big-project");
              }}
            />
          </div>
        </Space>
      </div>
    </div>
  );
}
