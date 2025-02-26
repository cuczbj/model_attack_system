"use client";

// import React, { useCallback, useState, useEffect } from "react";
// import { useDropzone } from "react-dropzone";
// import Box from "@mui/material/Box";
// import { styled } from "@mui/material/styles";
// import Button from "@mui/material/Button";
// import Typography from "@mui/material/Typography";
// import List from "@mui/material/List";
// import ListItem from "@mui/material/ListItem";
// import Divider from "@mui/material/Divider";
// import { useTheme } from "@mui/material/styles";
// 1. React 和外部库导入
import React, { useCallback, useState, useEffect } from "react";

// 2. 第三方库导入
import { useDropzone } from "react-dropzone";

// 3. MUI 组件导入
import { Box, Button, Typography, List, ListItem, Divider } from "@mui/material";

// 4. 样式和主题导入
import { styled, useTheme } from "@mui/material/styles";

// 自定义样式的上传区域
const AppReactDropzone = styled(Box)(({ theme }) => ({
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(4),
  textAlign: "center",
  color: theme.palette.text.secondary,
  transition: "border 0.3s ease-in-out",
  height: "200px", // 设置高度
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  "&:hover": {
    borderColor: theme.palette.primary.dark,
  },
}));

const FileUpload = () => {
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [recentFiles, setRecentFiles] = useState<string[]>([]);

  // 初始化时从 localStorage 获取最近上传记录
  useEffect(() => {
    const storedFiles = localStorage.getItem("recentFiles");

    if (storedFiles) {
      setRecentFiles(JSON.parse(storedFiles));
    }
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    // 获取文件名并更新状态
    const fileNames = acceptedFiles.map((file) => file.name);

    // 更新最近上传文件记录
    const updatedRecentFiles = [...recentFiles, ...fileNames].slice(-5); // 只保留最近 5 个
    
    setRecentFiles(updatedRecentFiles);

    // 保存到 localStorage
    localStorage.setItem("recentFiles", JSON.stringify(updatedRecentFiles));

    setUploadMessage("文件已成功处理！");
  }, [recentFiles]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  // 清空最近文件记录
  const clearRecentFiles = () => {
    setRecentFiles([]);
    localStorage.removeItem("recentFiles");
  };

  return (
    <div>
      <AppReactDropzone {...getRootProps()}>
        <input {...getInputProps()} />
        {isDragActive ? (
          <p style={{ fontWeight: "bold", color: "#1976d2" }}>
            释放文件到这里...
          </p>
        ) : (
          <p style={{ fontWeight: "bold" }}>
            拖放文件到这里，或点击选择上传模型文件
          </p>
        )}
      </AppReactDropzone>
      {/* 上传状态显示 */}
      {uploadMessage && (
        <Typography
          style={{ marginTop: "20px", color: "#1976d2", fontWeight: "bold" }}
        >
          {uploadMessage}
        </Typography>
      )}
      {/* 最近上传文件记录 */}
      {recentFiles.length > 0 && (
        <Box style={{ marginTop: "30px" }}>
          <Typography variant="h6" gutterBottom>
            最近上传的文件：
          </Typography>
          <List>
            {recentFiles.map((file, index) => (
              <React.Fragment key={index}>
                <ListItem>{file}</ListItem>
                {index < recentFiles.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
          <Button
            variant="contained"
            color="secondary"
            style={{ marginTop: "10px" }}
            onClick={clearRecentFiles}
          >
            清空记录
          </Button>
        </Box>
      )}
    </div>
  );
};

const HelpSection = () => {
  const theme = useTheme(); // 获取当前主题
  const isDarkMode = theme.palette.mode === "dark";

  return (
    <Box
      style={{
        marginTop: "30px",
        padding: "20px",
        borderRadius: "8px",
        backgroundColor: isDarkMode ? "#424242" : "#f5f5f5",
        color: isDarkMode ? "#ffffff" : "#000000",
      }}
    >
      <Typography variant="h6" gutterBottom>
        使用指南：
      </Typography>
      <Typography variant="body1" gutterBottom>
        - 将您的模型文件拖放到上方的区域，或点击选择文件进行上传。
      </Typography>
      <Typography variant="body1" gutterBottom>
        - 上传完成后，您可以在下方查看最近上传的文件记录。
      </Typography>
    </Box>
  );
};

export default function Page() {
  return (
    <div>
      <h1>机器学习模型攻击系统</h1>
      <div style={{ marginTop: "30px", marginBottom: "30px" }}></div>
      <FileUpload />
      <HelpSection />
    </div>
  );
}
