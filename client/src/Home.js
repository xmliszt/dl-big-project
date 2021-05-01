import { Upload, message, Button, Spin, Modal, Space, Progress } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import "./Home.css";
import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [uploadDone, setUploadDone] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadFilename, setUploadFilename] = useState("");
  const [showResult, setShowResult] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [fileList, setFileList] = useState([]);

  const props = {
    name: "audio",
    multiple: false,
    maxCount: 1,
    action: "https://genrewiz.herokuapp.com/upload",
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
        setUploadDone(true);
        message.success(`${info.file.name} file uploaded successfully.`);
        setUploadFilename(info.file.name);
      } else if (status === "error") {
        message.error(`${info.file.name} file upload failed.`);
      } else if (status === "removed") {
        message.warning(`${info.file.name} file removed successfully.`);
        setUploadDone(false);
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

  const submitUpload = async () => {
    try {
      setLoading(true);
      let response = await axios.post(
        "https://genrewiz.herokuapp.com/predict",
        {
          name: uploadFilename,
        }
      );
      var predictions = response.data.data.prediction;
      setPredictions(predictions);
      setLoading(false);
      setShowResult(true);
    } catch (err) {
      setLoading(false);
      console.error(err);
      message.error(err);
    }
  };

  const handleCloseResult = () => {
    setShowResult(false);
    setFileList([]);
    setUploadDone(false);
  };

  var UploadButton;
  if (uploadDone) {
    UploadButton = (
      <div className="upload-button-wrapper">
        <Button type="primary" size="large" onClick={submitUpload}>
          Predict Genres
        </Button>
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
      <Spin
        spinning={loading}
        tip="AI is predicting the genres for you..."
        size="large"
      >
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

          {UploadButton}
        </div>
      </Spin>
    </div>
  );
}
