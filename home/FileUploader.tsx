"use client";

import React, { useCallback } from 'react';

import { useDropzone } from 'react-dropzone';
import Box from '@mui/material/Box';
import { styled } from '@mui/material/styles';

// 自定义样式的上传区域
const AppReactDropzone = styled(Box)(({ theme }) => ({
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(4),
  textAlign: 'center',
  color: theme.palette.text.secondary,
  transition: 'border 0.3s ease-in-out',
  '&:hover': {
    borderColor: theme.palette.primary.dark,
  },
}));

const FileUpload = () => {
  const onDrop = useCallback((acceptedFiles:any) => {
    // 处理上传的文件
    console.log(acceptedFiles);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <AppReactDropzone {...getRootProps()}>
      <input {...getInputProps()} />
      {isDragActive ? (
        <p style={{ fontWeight: 'bold', color: '#1976d2' }}>释放文件到这里...</p>
      ) : (
        <p style={{ fontWeight: 'bold' }}>拖放文件到这里，或点击选择上传模型文件</p>
      )}
    </AppReactDropzone>
  );
};

export default FileUpload;
