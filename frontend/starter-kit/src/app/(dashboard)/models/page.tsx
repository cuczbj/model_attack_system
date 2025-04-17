"use client";
import ModelCreationPage from "./ModelCreationPage";
import React, { useState, useEffect, useCallback } from "react";
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Divider,
  Card,
  CardContent,
  CardActions,
  Chip,
  Slider,
  LinearProgress,
  Alert,
  AlertTitle,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Snackbar,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from "@mui/material";
import { styled } from "@mui/material/styles";
import { useDropzone } from "react-dropzone";

// 图标导入
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import SettingsIcon from "@mui/icons-material/Settings";
import ListIcon from "@mui/icons-material/List";
import RefreshIcon from "@mui/icons-material/Refresh";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import AutorenewIcon from "@mui/icons-material/Autorenew";
import SearchIcon from "@mui/icons-material/Search";
import AddIcon from "@mui/icons-material/Add";
import CloseIcon from "@mui/icons-material/Close";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";

// API基础URL
const API_URL = "http://localhost:5000";

// 数据类型定义
interface ModelType {
  [key: string]: string; // 模型名称 -> 模型类
}

interface ModelConfig {
  model_name: string;
  param_file: string;
  class_num: number;
  input_shape: [number, number];
  model_type: string;
  created_time: number;
  model_id?: string;
}

interface SearchResult {
  model_name: string;
  structure_found: boolean;
  structure_file?: string;
  parameters_found: boolean;
  matching_parameters: string[];
  id: string | null;
  auto_config: ModelConfig | null;
  missing_files?: {
    structure: boolean;
    parameters: boolean;
  };
}

interface PendingUpload {
  modelName: string;
  needStructure: boolean;
  needParams: boolean;
}

// 自定义样式组件
const StyledDropzone = styled(Box)(({ theme }) => ({
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(4),
  textAlign: "center",
  color: theme.palette.text.secondary,
  transition: "border 0.3s ease-in-out",
  height: "200px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  "&:hover": {
    borderColor: theme.palette.primary.dark,
  },
}));

const TabPanel = (props: {
  children?: React.ReactNode;
  index: number;
  value: number;
}) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

// API 请求函数
const fetchAvailableModels = async (): Promise<ModelType> => {
  try {
    const response = await fetch(`${API_URL}/api/models/available`);
    if (!response.ok) throw new Error("Failed to fetch available models");
    return await response.json();
  } catch (error) {
    console.error("Error fetching available models:", error);
    return {};
  }
};

const fetchModelParameters = async (): Promise<string[]> => {
  try {
    const response = await fetch(`${API_URL}/api/models/parameters`);
    if (!response.ok) throw new Error("Failed to fetch model parameters");
    return await response.json();
  } catch (error) {
    console.error("Error fetching model parameters:", error);
    return [];
  }
};

const fetchModelConfigurations = async (): Promise<ModelConfig[]> => {
  try {
    const response = await fetch(`${API_URL}/api/models/configurations`);
    if (!response.ok) throw new Error("Failed to fetch model configurations");
    return await response.json();
  } catch (error) {
    console.error("Error fetching model configurations:", error);
    return [];
  }
};

const triggerModelScan = async (): Promise<any> => {
  try {
    const response = await fetch(`${API_URL}/api/models/scan`, {
      method: "POST",
    });
    if (!response.ok) throw new Error("Failed to trigger model scan");
    return await response.json();
  } catch (error) {
    console.error("Error triggering model scan:", error);
    throw error;
  }
};

const createModelConfiguration = async (config: Partial<ModelConfig>): Promise<any> => {
  try {
    const response = await fetch(`${API_URL}/api/models/configurations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
    });
    
    const data = await response.json();
    
    if (!response.ok) throw new Error(data.error || "Failed to create model configuration");
    return data;
  } catch (error) {
    console.error("Error creating model configuration:", error);
    throw error;
  }
};

const deleteModelConfiguration = async (modelName: string, paramFile: string): Promise<any> => {
  try {
    const response = await fetch(`${API_URL}/api/models/configurations/${modelName}/${paramFile}`, {
      method: "DELETE",
    });
    
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "Failed to delete model configuration");
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error deleting model configuration:", error);
    throw error;
  }
};

const autoDetectModelConfig = async (paramFile: string): Promise<any> => {
  try {
    const response = await fetch(`${API_URL}/api/models/auto-detect-config`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ param_file: paramFile }),
    });
    
    const data = await response.json();
    
    if (!response.ok) throw new Error(data.error || "Failed to auto-detect model configuration");
    return data;
  } catch (error) {
    console.error("Error auto-detecting model configuration:", error);
    throw error;
  }
};

const validateModelConfig = async (config: Partial<ModelConfig>): Promise<any> => {
  try {
    const response = await fetch(`${API_URL}/api/models/validate-config`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
    });
    
    const data = await response.json();
    return {
      valid: response.ok,
      message: data.message || (response.ok ? "配置有效" : "配置无效"),
      data
    };
  } catch (error) {
    console.error("Error validating model configuration:", error);
    return {
      valid: false,
      message: "验证配置时出错",
      error
    };
  }
};

const searchModelByName = async (modelName: string): Promise<SearchResult> => {
  try {
    const response = await fetch(`${API_URL}/api/models/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model_name: modelName }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "搜索失败");
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error searching for model:", error);
    throw error;
  }
};

// 模型搜索组件
const ModelSearch = ({ onSearchResult, initialSearchTerm = "" }) => {
  const [modelName, setModelName] = useState(initialSearchTerm);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState(null);
  
  // 当initialSearchTerm变化时更新modelName
  useEffect(() => {
    if (initialSearchTerm) {
      setModelName(initialSearchTerm);
    }
  }, [initialSearchTerm]);
  
  const handleSearch = async () => {
    if (!modelName.trim()) {
      setError("请输入模型名称");
      return;
    }
    
    setSearching(true);
    setError(null);
    
    try {
      const result = await searchModelByName(modelName);
      onSearchResult(result);
    } catch (error) {
      console.error("Search error:", error);
      setError(error.message || "搜索过程中发生错误");
    } finally {
      setSearching(false);
    }
  };
  
  return (
    <Box sx={{ mb: 4 }}>
      <Typography variant="h6" gutterBottom>
        搜索模型
      </Typography>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        输入模型名称，系统将自动检查是否已存在对应的模型结构和参数文件
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={8}>
          <TextField
            fullWidth
            label="模型名称"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="例如: MLP, VGG16, IR152, FaceNet64"
            error={!!error}
            helperText={error}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleSearch();
              }
            }}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <Button
            fullWidth
            variant="contained"
            onClick={handleSearch}
            disabled={searching || !modelName.trim()}
            sx={{ height: '56px' }}
            startIcon={searching ? <CircularProgress size={24} color="inherit" /> : <SearchIcon />}
          >
            {searching ? "搜索中..." : "搜索模型"}
          </Button>
        </Grid>
      </Grid>
    </Box>
  );
};

// 搜索结果组件
const SearchResult = ({ result, onCreateConfig, onUpload }) => {
  if (!result) return null;
  
  // 确定具体缺少哪些文件
  const missingStructure = !result.structure_found;
  const missingParameters = !result.parameters_found;
  
  return (
    <Card variant="outlined" sx={{ mb: 4 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          搜索结果: {result.model_name}
        </Typography>
        
        <Grid container spacing={2}>
          {/* 模型结构状态提示 */}
          <Grid item xs={12} md={6}>
            <Alert 
              severity={result.structure_found ? "success" : "warning"}
              icon={result.structure_found ? <CheckCircleIcon /> : <ErrorIcon />}
              sx={{ mb: 2 }}
            >
              <AlertTitle>{result.structure_found ? "模型结构已找到" : "模型结构未找到"}</AlertTitle>
              {result.structure_found 
                ? `系统已找到 ${result.model_name} 的模型结构定义` 
                : `需要上传 ${result.model_name}.py 模型结构文件`}
            </Alert>
          </Grid>
          
          {/* 模型参数状态提示 */}
          <Grid item xs={12} md={6}>
            <Alert 
              severity={result.parameters_found ? "success" : "warning"}
              icon={result.parameters_found ? <CheckCircleIcon /> : <ErrorIcon />}
              sx={{ mb: 2 }}
            >
              <AlertTitle>{result.parameters_found ? "模型参数已找到" : "模型参数未找到"}</AlertTitle>
              {result.parameters_found 
                ? `找到 ${result.matching_parameters.length} 个匹配的参数文件` 
                : `需要上传 ${result.model_name}_xxx.pth/.tar/.pkl 格式的参数文件`}
            </Alert>
          </Grid>
        </Grid>
        
        {/* 匹配的参数文件列表 */}
        {result.matching_parameters.length > 0 && (
          <Box sx={{ mt: 2, mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              匹配的参数文件:
            </Typography>
            <List dense>
              {result.matching_parameters.map((param, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText primary={param} />
                </ListItem>
              ))}
            </List>
          </Box>
        )}
        
        {/* 操作按钮区域 */}
        <Box sx={{ mt: 2 }}>
          {result.id ? (
            <Alert severity="info">
              该模型已配置，ID: {result.id}
            </Alert>
          ) : result.structure_found && result.parameters_found ? (
            <Button
              variant="contained"
              color="primary"
              onClick={() => onCreateConfig(result.auto_config)}
              startIcon={<CheckCircleIcon />}
            >
              自动创建配置
            </Button>
          ) : (
            <Box>
              <Typography variant="subtitle1" color="error" gutterBottom>
                缺少必要文件，请上传：
              </Typography>
              
              {/* 缺少模型结构文件时显示上传提示 */}
              {missingStructure && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  <AlertTitle>缺少模型结构文件</AlertTitle>
                  请上传名为 <strong>{result.model_name}.py</strong> 的模型结构文件
                </Alert>
              )}
              
              {/* 缺少参数文件时显示上传提示 */}
              {missingParameters && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  <AlertTitle>缺少模型参数文件</AlertTitle>
                  请上传形如 <strong>{result.model_name}_xxxx.pth</strong> 或 <strong>{result.model_name}_xxxx.tar</strong> 的参数文件
                </Alert>
              )}
              
              <Button
                variant="contained"
                color="warning"
                onClick={() => onUpload(result.model_name, missingStructure, missingParameters)}
                startIcon={<CloudUploadIcon />}
                sx={{ mt: 1 }}
              >
                前往上传缺失文件
              </Button>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

// 文件上传组件
const ModelUploader = ({ onUploadSuccess, searchModelName }) => {
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [recentUploads, setRecentUploads] = useState<string[]>([]);
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(null);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "info" | "warning";
  }>({
    open: false,
    message: "",
    severity: "info",
  });

  // 加载最近上传记录和待上传信息
  useEffect(() => {
    // 加载最近上传记录
    const saved = localStorage.getItem("recentModelUploads");
    if (saved) {
      setRecentUploads(JSON.parse(saved));
    }
    
    // 加载待上传信息
    const pendingData = localStorage.getItem('pendingUpload');
    if (pendingData) {
      try {
        setPendingUpload(JSON.parse(pendingData));
      } catch (e) {
        console.error("解析待上传数据失败", e);
      }
    }
  }, []);

  // 上传成功后清除待上传信息
  const handleUploadComplete = useCallback(() => {
    localStorage.removeItem('pendingUpload');
    setPendingUpload(null);
    
    // 通知外部组件上传成功
    onUploadSuccess();
  }, [onUploadSuccess]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    const formData = new FormData();
    formData.append("file", file);

    // 重置状态并开始进度
    setUploadStatus("uploading");
    setUploadProgress(0);

    // 创建请求
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/checkpoint`, true);

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const progress = Math.round((event.loaded / event.total) * 100);
        setUploadProgress(progress);
      }
    };

    xhr.onload = () => {
      if (xhr.status === 200) {
        try {
          const response = JSON.parse(xhr.responseText);
          setUploadStatus("success");
          
          // 添加到最近上传记录
          const newRecentUploads = [file.name, ...recentUploads].slice(0, 5);
          setRecentUploads(newRecentUploads);
          localStorage.setItem("recentModelUploads", JSON.stringify(newRecentUploads));
          
          // 显示通知
          if (response.auto_detection && response.auto_detection.detected_model) {
            setNotification({
              open: true,
              message: `上传成功！检测到匹配的模型类型: ${response.auto_detection.detected_model}`,
              severity: "success",
            });
          } else {
            setNotification({
              open: true,
              message: "文件上传成功！",
              severity: "success",
            });
          }
          
          // 通知外部组件上传成功
          onUploadSuccess();
          
          // 重置进度
          setTimeout(() => {
            setUploadProgress(0);
            setUploadStatus(null);
          }, 3000);
        } catch (error) {
          setUploadStatus("error");
          setNotification({
            open: true,
            message: "解析响应时出错",
            severity: "error",
          });
        }
      } else {
        setUploadStatus("error");
        setNotification({
          open: true,
          message: "上传失败，服务器返回错误",
          severity: "error",
        });
      }
    };

    xhr.onerror = () => {
      setUploadStatus("error");
      setNotification({
        open: true,
        message: "上传失败，网络错误",
        severity: "error",
      });
    };

    xhr.send(formData);
  }, [recentUploads, onUploadSuccess]);

  // 确定接受的文件类型
  const fileTypes = {
    'application/octet-stream': ['.pth', '.pkl', '.tar'],
    'text/x-python': ['.py'],
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: fileTypes,
    maxFiles: 1,
  });

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        上传模型文件
      </Typography>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        支持的文件格式:
        <br />- 模型结构文件: .py
        <br />- 模型参数文件: .pth, .pkl, .tar
      </Typography>
      
      {/* 如果有待上传的文件需求，显示特定提示 */}
      {pendingUpload && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <AlertTitle>需要上传 {pendingUpload.modelName} 的文件</AlertTitle>
          <Typography variant="body2">
            {pendingUpload.needStructure && (
              <Box sx={{ mb: 1 }}>
                • 模型结构文件: <strong>{pendingUpload.modelName}.py</strong>
              </Box>
            )}
            {pendingUpload.needParams && (
              <Box>
                • 模型参数文件: <strong>{pendingUpload.modelName}_xxxx.pth/tar/pkl</strong>
              </Box>
            )}
          </Typography>
          <Button 
            size="small" 
            onClick={() => {
              localStorage.removeItem('pendingUpload');
              setPendingUpload(null);
            }} 
            sx={{ mt: 1 }}
            color="inherit"
          >
            取消此操作
          </Button>
        </Alert>
      )}
      
      <Alert severity="info" sx={{ mb: 2 }}>
        <AlertTitle>命名规范</AlertTitle>
        <Typography variant="body2">
          <strong>模型结构文件</strong>应遵循 <code>[ModelName].py</code> 格式，例如 <code>MLP.py</code>、<code>VGG16.py</code>
          <br />
          <strong>参数文件</strong>应遵循 <code>[ModelName]_[Details].ext</code> 格式，例如 <code>MLP_ATT40.pkl</code>、<code>VGG16_CelebA1000.tar</code>
        </Typography>
      </Alert>
      
      <StyledDropzone {...getRootProps()}>
        <input {...getInputProps()} />
        <CloudUploadIcon style={{ fontSize: 50, marginBottom: 16 }} color="primary" />
        
        {isDragActive ? (
          <Typography variant="body1">释放文件到这里上传...</Typography>
        ) : (
          <Typography variant="body1">
            拖放文件到这里，或点击选择文件
          </Typography>
        )}
      </StyledDropzone>
      
      {/* 进度指示器 */}
      {uploadStatus === "uploading" && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" gutterBottom>
            上传进度: {uploadProgress}%
          </Typography>
          <LinearProgress variant="determinate" value={uploadProgress} />
        </Box>
      )}
      
      {/* 状态提示 */}
      {uploadStatus === "success" && (
        <Alert severity="success" sx={{ mt: 2 }}>
          文件上传成功！
        </Alert>
      )}
      
      {uploadStatus === "error" && (
        <Alert severity="error" sx={{ mt: 2 }}>
          上传失败，请检查服务器连接或文件格式。
        </Alert>
      )}
      
      {/* 上传成功后，如果有待完成的模型配置，提示用户继续搜索该模型 */}
      {uploadStatus === "success" && pendingUpload && (
        <Alert severity="success" sx={{ mt: 2 }}>
          <AlertTitle>文件上传成功！</AlertTitle>
          <Typography variant="body2">
            请再次搜索 <strong>{pendingUpload.modelName}</strong> 以检查是否已满足创建条件。
          </Typography>
          <Button 
            variant="outlined" 
            size="small" 
            sx={{ mt: 1 }}
            onClick={() => {
              // 清除pendingUpload
              handleUploadComplete();
              // 重新搜索该模型
              searchModelName(pendingUpload.modelName);
            }}
          >
            重新检查该模型
          </Button>
        </Alert>
      )}
      
      {/* 最近上传列表 */}
      {recentUploads.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="subtitle1" gutterBottom>
            最近上传的文件
          </Typography>
          
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>文件名</TableCell>
                  <TableCell align="right">操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {recentUploads.map((filename, index) => (
                  <TableRow key={index}>
                    <TableCell>{filename}</TableCell>
                    <TableCell align="right">
                      <Button 
                        size="small" 
                        onClick={async () => {
                          try {
                            const result = await autoDetectModelConfig(filename);
                            setNotification({
                              open: true,
                              message: result.message,
                              severity: "info",
                            });
                          } catch (error) {
                            setNotification({
                              open: true,
                              message: "自动检测配置失败",
                              severity: "error",
                            });
                          }
                        }}
                      >
                        自动检测
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          <Button 
            variant="outlined" 
            size="small"
            sx={{ mt: 1 }}
            onClick={() => {
              setRecentUploads([]);
              localStorage.removeItem("recentModelUploads");
            }}
          >
            清空记录
          </Button>
        </Box>
      )}
      
      {/* 通知 */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
      >
        <Alert onClose={handleCloseNotification} severity={notification.severity}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

// 模型配置组件
const ModelConfiguration = ({ onConfigCreated, refreshData }) => {
  // 状态
  const [availableModels, setAvailableModels] = useState<ModelType>({});
  const [modelParameters, setModelParameters] = useState<string[]>([]);
  const [modelConfigurations, setModelConfigurations] = useState<ModelConfig[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  
  // 表单状态
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedParameter, setSelectedParameter] = useState<string>("");
  const [classNum, setClassNum] = useState<number>(40);
  const [inputShape, setInputShape] = useState<[number, number]>([112, 92]);
  const [validationResult, setValidationResult] = useState<{
    valid: boolean;
    message: string;
  } | null>(null);
  const [isCreating, setIsCreating] = useState<boolean>(false);
  
  // 加载数据
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [models, parameters, configs] = await Promise.all([
        fetchAvailableModels(),
        fetchModelParameters(),
        fetchModelConfigurations(),
      ]);
      
      setAvailableModels(models);
      setModelParameters(parameters);
      setModelConfigurations(configs);
      
      // 如果数据加载后没有选择模型，设置默认值
      if (selectedModel === "" && Object.keys(models).length > 0) {
        setSelectedModel(Object.keys(models)[0]);
      }
      
      if (selectedParameter === "" && parameters.length > 0) {
        setSelectedParameter(parameters[0]);
      }
    } catch (error) {
      console.error("Failed to load model data:", error);
    } finally {
      setLoading(false);
    }
  }, [selectedModel, selectedParameter]);
  
  // 首次加载和刷新数据
  useEffect(() => {
    loadData();
  }, [loadData]);
  
  // 当用户选择模型时，自动设置默认的类别数和输入形状
  useEffect(() => {
    if (selectedModel === "MLP") {
      setClassNum(40);
      setInputShape([112, 92]);
    } else if (["VGG16", "FaceNet64", "IR152"].includes(selectedModel)) {
      setClassNum(1000);
      setInputShape([64, 64]);
    }
  }, [selectedModel]);
  
  // 验证配置
  const validateConfig = async () => {
    if (!selectedModel || !selectedParameter) {
      setValidationResult({
        valid: false,
        message: "请选择模型和参数文件"
      });
      return false;
    }
    
    try {
      const config = {
        model_name: selectedModel,
        param_file: selectedParameter,
        class_num: classNum,
        input_shape: inputShape,
      };
      
      const result = await validateModelConfig(config);
      setValidationResult(result);
      return result.valid;
    } catch (error) {
      setValidationResult({
        valid: false,
        message: "验证配置时出错"
      });
      return false;
    }
  };
  
  // 创建配置
  const createConfig = async () => {
    try {
      setIsCreating(true);
      
      // 首先验证配置
      const isValid = await validateConfig();
      if (!isValid) return;
      
      // 创建配置
      const config = {
        model_name: selectedModel,
        param_file: selectedParameter,
        class_num: classNum,
        input_shape: inputShape,
        model_type: selectedModel,
      };
      
      await createModelConfiguration(config);
      
      // 重新加载数据
      await loadData();
      
      // 通知父组件
      onConfigCreated();
    } catch (error) {
      console.error("Failed to create model configuration:", error);
    } finally {
      setIsCreating(false);
    }
  };
  
  // 处理模型选择变化
  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
    // 重置验证结果
    setValidationResult(null);
  };
  
  // 处理参数选择变化
  const handleParameterChange = (event) => {
    setSelectedParameter(event.target.value);
    // 重置验证结果
    setValidationResult(null);
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          创建模型配置
        </Typography>
        
        <Button 
          startIcon={<RefreshIcon />}
          onClick={() => {
            refreshData();
            loadData();
          }}
        >
          刷新
        </Button>
      </Box>
      
      {loading ? (
        <LinearProgress />
      ) : (
        <Grid container spacing={3}>
          {/* 模型选择 */}
          <Grid item xs={12} md={6}>
            <FormControl fullWidth margin="normal">
              <InputLabel id="model-select-label">模型类型</InputLabel>
              <Select
                labelId="model-select-label"
                value={selectedModel}
                label="模型类型"
                onChange={handleModelChange}
              >
                {Object.entries(availableModels).map(([key, value]) => (
                  <MenuItem key={key} value={key}>
                    {key} ({value})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          {/* 参数文件选择 */}
          <Grid item xs={12} md={6}>
            <FormControl fullWidth margin="normal">
              <InputLabel id="parameter-select-label">参数文件</InputLabel>
              <Select
                labelId="parameter-select-label"
                value={selectedParameter}
                label="参数文件"
                onChange={handleParameterChange}
              >
                {modelParameters.map((param) => (
                  <MenuItem key={param} value={param}>
                    {param}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          {/* 类别数 */}
          <Grid item xs={12} md={6}>
            <TextField
              label="分类数量"
              type="number"
              fullWidth
              margin="normal"
              value={classNum}
              onChange={(e) => setClassNum(parseInt(e.target.value) || 0)}
              InputProps={{
                inputProps: { min: 1 }
              }}
              helperText={(selectedModel === "MLP") ? 
                "AT&T Faces 数据集默认 40 类" : 
                (["VGG16", "FaceNet64", "IR152"].includes(selectedModel) ? 
                  "CelebA 数据集默认 1000 类" : 
                  "请设置分类数量")}
            />
          </Grid>
          
          {/* 输入形状 */}
          <Grid item xs={12} md={6}>
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                输入形状:
              </Typography>
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <TextField
                    label="高度"
                    type="number"
                    fullWidth
                    value={inputShape[0]}
                    onChange={(e) => setInputShape([parseInt(e.target.value) || 0, inputShape[1]])}
                    InputProps={{
                      inputProps: { min: 1 }
                    }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="宽度"
                    type="number"
                    fullWidth
                    value={inputShape[1]}
                    onChange={(e) => setInputShape([inputShape[0], parseInt(e.target.value) || 0])}
                    InputProps={{
                      inputProps: { min: 1 }
                    }}
                  />
                </Grid>
              </Grid>
              <Typography variant="caption" color="textSecondary">
                {selectedModel === "MLP" ? "MLP模型默认输入形状为 112×92" : 
                 ["VGG16", "FaceNet64", "IR152"].includes(selectedModel) ? 
                 "图像模型默认输入形状为 64×64" : "请设置合适的输入形状"}
              </Typography>
            </Box>
          </Grid>
          
          {/* 验证和创建按钮 */}
          <Grid item xs={12}>
            <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
              <Button
                variant="outlined"
                color="primary"
                onClick={validateConfig}
                disabled={!selectedModel || !selectedParameter || isCreating}
              >
                验证配置
              </Button>
              
              <Button
                variant="contained"
                color="primary"
                onClick={createConfig}
                disabled={!selectedModel || !selectedParameter || isCreating}
              >
                {isCreating ? <CircularProgress size={24} /> : "创建配置"}
              </Button>
            </Box>
          </Grid>
          
          {/* 验证结果 */}
          {validationResult && (
            <Grid item xs={12}>
              <Alert 
                severity={validationResult.valid ? "success" : "error"}
                sx={{ mt: 2 }}
              >
                {validationResult.message}
              </Alert>
            </Grid>
          )}
        </Grid>
      )}
      
      {/* 当前配置列表 */}
      {modelConfigurations.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            现有模型配置
          </Typography>
          
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>模型类型</TableCell>
                  <TableCell>参数文件</TableCell>
                  <TableCell>分类数</TableCell>
                  <TableCell>输入形状</TableCell>
                  <TableCell align="right">操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {modelConfigurations.map((config) => (
                  <TableRow key={`${config.model_name}-${config.param_file}`}>
                    <TableCell>{config.model_name}</TableCell>
                    <TableCell>{config.param_file}</TableCell>
                    <TableCell>{config.class_num}</TableCell>
                    <TableCell>{config.input_shape.join(' × ')}</TableCell>
                    <TableCell align="right">
                      <IconButton 
                        size="small" 
                        color="error"
                        onClick={async () => {
                          if (window.confirm(`确定要删除 ${config.model_name} + ${config.param_file} 的配置吗？`)) {
                            try {
                              await deleteModelConfiguration(config.model_name, config.param_file);
                              loadData(); // 重新加载数据
                            } catch (error) {
                              console.error("Failed to delete configuration:", error);
                              alert("删除配置失败");
                            }
                          }
                        }}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}
    </Box>
  );
};

// 模型管理组件
const ModelManager = ({ refreshData }) => {
  const [modelConfigurations, setModelConfigurations] = useState<ModelConfig[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  
  // 加载数据
  const loadData = async () => {
    setLoading(true);
    try {
      const configs = await fetchModelConfigurations();
      setModelConfigurations(configs);
    } catch (error) {
      console.error("Failed to load model configurations:", error);
    } finally {
      setLoading(false);
    }
  };
  
  // 首次加载
  useEffect(() => {
    loadData();
  }, []);
  
  // 删除配置
  const handleDeleteConfig = async (modelName: string, paramFile: string) => {
    if (!window.confirm(`确定要删除 ${modelName} + ${paramFile} 的配置吗？`)) {
      return;
    }
    
    try {
      await deleteModelConfiguration(modelName, paramFile);
      loadData(); // 重新加载数据
      refreshData(); // 通知父组件刷新
    } catch (error) {
      console.error("Failed to delete configuration:", error);
      alert("删除配置失败");
    }
  };
  
  // 测试配置
  const handleTestConfig = async (config: ModelConfig) => {
    try {
      const result = await validateModelConfig({
        model_name: config.model_name,
        param_file: config.param_file,
        class_num: config.class_num,
      });
      
      alert(result.message);
    } catch (error) {
      console.error("Failed to test configuration:", error);
      alert("测试配置失败");
    }
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          模型管理
        </Typography>
        
        <Button 
          startIcon={<RefreshIcon />}
          onClick={() => {
            loadData();
            refreshData();
          }}
        >
          刷新
        </Button>
      </Box>
      
      {loading ? (
        <LinearProgress />
      ) : modelConfigurations.length === 0 ? (
        <Alert severity="info" sx={{ mt: 2 }}>
          <AlertTitle>没有配置</AlertTitle>
          还没有创建模型配置，请在"模型配置"标签页中创建
        </Alert>
      ) : (
        <Grid container spacing={2}>
          {modelConfigurations.map((config) => (
            <Grid item xs={12} md={6} key={`${config.model_name}-${config.param_file}`}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {config.model_name}
                    {config.model_id && (
                      <Typography variant="caption" color="textSecondary" sx={{ ml: 1 }}>
                        ID: {config.model_id}
                      </Typography>
                    )}
                  </Typography>
                  
                  <Typography color="textSecondary" gutterBottom>
                    参数文件: {config.param_file}
                  </Typography>
                  
                  <Box sx={{ mt: 2 }}>
                    <Chip 
                      label={`${config.class_num} 类`}
                      color="primary"
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    
                    <Chip 
                      label={`输入: ${config.input_shape.join(' × ')}`}
                      color="secondary"
                      size="small"
                    />
                  </Box>
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" color="textSecondary">
                      创建时间: {new Date(config.created_time * 1000).toLocaleString()}
                    </Typography>
                  </Box>
                </CardContent>
                
                <CardActions>
                  <Button 
                    size="small" 
                    startIcon={<CheckCircleIcon />}
                    onClick={() => handleTestConfig(config)}
                  >
                    测试
                  </Button>
                  
                  <Button 
                    size="small" 
                    color="error"
                    startIcon={<DeleteIcon />}
                    onClick={() => handleDeleteConfig(config.model_name, config.param_file)}
                  >
                    删除
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
};

// 主组件
const ModelManagementPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [refreshCounter, setRefreshCounter] = useState(0);
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [searchModelTerm, setSearchModelTerm] = useState("");
  const [showModelCreation, setShowModelCreation] = useState(false);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "info" | "warning";
  }>({
    open: false,
    message: "",
    severity: "info",
  });

  // 切换标签
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // 触发刷新
  const refreshData = useCallback(async () => {
    try {
      await triggerModelScan();
      setRefreshCounter(prev => prev + 1);
      setNotification({
        open: true,
        message: "模型数据已刷新",
        severity: "success",
      });
    } catch (error) {
      console.error("Failed to refresh data:", error);
      setNotification({
        open: true,
        message: "刷新数据失败",
        severity: "error",
      });
    }
  }, []);

  // 处理上传成功
  const handleUploadSuccess = useCallback(() => {
    // 自动刷新数据
    refreshData();
    
    // 如果有搜索结果，重新搜索以更新状态
    if (searchResult) {
      searchModelByName(searchResult.model_name)
        .then(newResult => setSearchResult(newResult))
        .catch(error => console.error("Error refreshing search:", error));
    }
  }, [refreshData, searchResult]);

  // 处理配置创建成功
  const handleConfigCreated = useCallback(() => {
    setNotification({
      open: true,
      message: "模型配置创建成功",
      severity: "success",
    });
    
    // 切换到管理标签
    setTabValue(2);
    
    // 刷新搜索结果
    if (searchResult) {
      searchModelByName(searchResult.model_name)
        .then(newResult => setSearchResult(newResult))
        .catch(error => console.error("Error refreshing search:", error));
    }
  }, [searchResult]);

  // 搜索模型
  const searchModel = useCallback(async (modelName: string) => {
    if (!modelName.trim()) return;
    
    setSearchModelTerm(modelName);
    
    try {
      const result = await searchModelByName(modelName);
      setSearchResult(result);
      
      // 根据结果显示不同通知
      if (result.id) {
        // 已配置
        setNotification({
          open: true,
          message: `模型 ${result.model_name} 已配置完成，ID: ${result.id}`,
          severity: "info",
        });
      } else if (result.structure_found && result.parameters_found) {
        // 找到结构和参数，但未配置
        setNotification({
          open: true,
          message: `找到 ${result.model_name} 的结构和参数文件，可以创建配置`,
          severity: "success",
        });
      } else {
        // 缺失组件
        setNotification({
          open: true,
          message: `${result.model_name} 配置不完整，需要上传缺失的文件`,
          severity: "warning",
        });
      }
    } catch (error) {
      console.error("Search error:", error);
      setNotification({
        open: true,
        message: `搜索失败: ${error.message || "未知错误"}`,
        severity: "error",
      });
    }
  }, []);

  // 处理搜索结果
  const handleSearchResult = useCallback((result: SearchResult) => {
    setSearchResult(result);
    
    // 根据结果显示不同通知
    if (result.id) {
      // 已配置
      setNotification({
        open: true,
        message: `模型 ${result.model_name} 已配置完成，ID: ${result.id}`,
        severity: "info",
      });
    } else if (result.structure_found && result.parameters_found) {
      // 找到结构和参数，但未配置
      setNotification({
        open: true,
        message: `找到 ${result.model_name} 的结构和参数文件，可以创建配置`,
        severity: "success",
      });
    } else {
      // 缺失组件
      setNotification({
        open: true,
        message: `${result.model_name} 配置不完整，需要上传缺失的文件`,
        severity: "warning",
      });
    }
  }, []);

  // 处理自动创建配置
  const handleAutoCreateConfig = useCallback(async (configData) => {
    if (!configData) return;
    
    try {
      await createModelConfiguration(configData);
      
      setNotification({
        open: true,
        message: "模型配置创建成功",
        severity: "success",
      });
      
      // 刷新数据并更新搜索结果
      await refreshData();
      
      if (searchResult) {
        const newResult = await searchModelByName(searchResult.model_name);
        setSearchResult(newResult);
      }
      
      // 切换到管理标签
      setTabValue(2);
    } catch (error) {
      setNotification({
        open: true,
        message: `创建配置失败: ${error.message}`,
        severity: "error",
      });
    }
  }, [refreshData, searchResult]);

  // 处理前往上传缺失文件
  const handleNavigateToUpload = useCallback((modelName, needStructure, needParams) => {
    // 保存需要上传的信息到localStorage，以便上传组件使用
    localStorage.setItem('pendingUpload', JSON.stringify({
      modelName,
      needStructure,
      needParams
    }));
    
    // 切换到上传标签页
    setTabValue(0);
    
    // 显示提示信息
    setNotification({
      open: true,
      message: `请上传${modelName}的${needStructure && needParams ? "结构文件和参数文件" : needStructure ? "结构文件" : "参数文件"}`,
      severity: "info",
    });
  }, []);

  // 处理模型创建完成
  const handleModelCreationComplete = useCallback((model) => {
    setShowModelCreation(false);
    setNotification({
      open: true,
      message: `模型 ${model?.model_name || ''} 创建成功！`,
      severity: "success",
    });
    refreshData();
  }, [refreshData]);

  // 关闭通知
  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  // 如果显示模型创建界面，则直接渲染ModelCreationPage
  if (showModelCreation) {
    return (
      <Box sx={{ width: '100%', mt: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Button
            startIcon={<ArrowBackIcon />}
            onClick={() => setShowModelCreation(false)}
          >
            返回模型管理
          </Button>
        </Box>
        
        <ModelCreationPage onComplete={handleModelCreationComplete} />
        
        {/* 全局通知 */}
        <Snackbar
          open={notification.open}
          autoHideDuration={6000}
          onClose={handleCloseNotification}
        >
          <Alert onClose={handleCloseNotification} severity={notification.severity}>
            {notification.message}
          </Alert>
        </Snackbar>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', mt: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        机器学习模型管理
      </Typography>
      
      {/* 创建新模型按钮 */}
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<AddIcon />}
          onClick={() => setShowModelCreation(true)}
          sx={{ px: 4, py: 1.5 }}
        >
          创建新模型
        </Button>
      </Box>
      
      {/* 模型搜索部分 */}
      <Box sx={{ mb: 4 }}>
        <ModelSearch 
          onSearchResult={handleSearchResult} 
          initialSearchTerm={searchModelTerm}
        />
        
        {searchResult && (
          <SearchResult 
            result={searchResult} 
            onCreateConfig={handleAutoCreateConfig} 
            onUpload={handleNavigateToUpload}
          />
        )}
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
        <Button 
          variant="outlined" 
          startIcon={<RefreshIcon />}
          onClick={refreshData}
        >
          刷新模型数据
        </Button>
      </Box>
      
      <Paper sx={{ width: '100%' }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          centered
        >
          <Tab icon={<CloudUploadIcon />} label="上传模型" />
          <Tab icon={<SettingsIcon />} label="模型配置" />
          <Tab icon={<ListIcon />} label="模型管理" />
        </Tabs>
        
        <TabPanel value={tabValue} index={0}>
          <ModelUploader 
            onUploadSuccess={handleUploadSuccess} 
            searchModelName={searchModel}
          />
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <ModelConfiguration 
            onConfigCreated={handleConfigCreated} 
            refreshData={refreshData}
            key={`config-${refreshCounter}`}
          />
        </TabPanel>
        
        <TabPanel value={tabValue} index={2}>
          <ModelManager 
            refreshData={refreshData}
            key={`manager-${refreshCounter}`}
          />
        </TabPanel>
      </Paper>
      
      <Box sx={{ mt: 4, p: 2, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
        <Typography variant="subtitle1" gutterBottom>
          使用指南:
        </Typography>
        <Typography variant="body2">
          1. 在顶部搜索框输入模型名称，检查系统中是否存在该模型的结构和参数文件。
        </Typography>
        <Typography variant="body2">
          2. 如果检测到模型结构和参数文件都存在，可以一键自动创建配置。
        </Typography>
        <Typography variant="body2">
          3. 如果缺少结构或参数文件，系统会提示您需要上传的具体文件，并引导您前往上传页面。
        </Typography>
        <Typography variant="body2">
          4. 所有文件上传完成后，再次搜索该模型，系统会自动检测文件是否齐全可创建配置。
        </Typography>
        <Typography variant="body2">
          5. 创建好的模型配置会在攻击效果评估页面的"选择预测模型"下拉栏中自动显示。
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
          提示: 请确保模型结构文件和参数文件遵循命名规范，这样系统才能正确匹配它们。
        </Typography>
      </Box>
      
      {/* 全局通知 */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
      >
        <Alert onClose={handleCloseNotification} severity={notification.severity}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ModelManagementPage;