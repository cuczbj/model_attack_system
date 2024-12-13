"use client";

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import Box from '@mui/material/Box';
import { styled } from '@mui/material/styles';
import axios from 'axios';

// 自定义样式的上传区域
const AppReactDropzone = styled(Box)(({ theme }) => ({
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(4),
  textAlign: 'center',
  color: theme.palette.text.secondary,
  transition: 'border 0.3s ease-in-out',
  height: '200px', // 设置高度
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  '&:hover': {
    borderColor: theme.palette.primary.dark,
  },
}));

const FileUpload = () => {
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const formData = new FormData();
    acceptedFiles.forEach((file) => {
      formData.append('file', file); // 将文件加入到 FormData
    });

    // 使用 Axios 将文件发送到后端
    axios
      .post('http://localhost:8080/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      .then((response:any) => {
        setUploadMessage('文件上传成功！');
        console.log('后端响应：', response.data);
      })
      .catch((error:any) => {
        setUploadMessage('文件上传失败，请稍后重试！');
        console.error('文件上传错误：', error);
      });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div>
      <AppReactDropzone {...getRootProps()}>
        <input {...getInputProps()} />
        {isDragActive ? (
          <p style={{ fontWeight: 'bold', color: '#1976d2' }}>释放文件到这里...</p>
        ) : (
          <p style={{ fontWeight: 'bold' }}>拖放文件到这里，或点击选择上传模型文件</p>
        )}
      </AppReactDropzone>
      {/* 上传状态显示 */}
      {uploadMessage && <p style={{ marginTop: '20px', color: '#1976d2' }}>{uploadMessage}</p>}
    </div>
  );
};

export default function Page() {
  return (
    <div>
      <h1>机器学习模型攻击系统</h1>
      <div style={{ marginTop: '30px', marginBottom: '30px' }}></div>
      <FileUpload />
    </div>
  );
}

