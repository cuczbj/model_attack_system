"use client";

import React, { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import TextField from "@mui/material/TextField";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormControlLabel from "@mui/material/FormControlLabel";
import Checkbox from "@mui/material/Checkbox";
import Divider from "@mui/material/Divider";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Tooltip from "@mui/material/Tooltip";
import IconButton from "@mui/material/IconButton";
import Alert from "@mui/material/Alert";
import Chip from "@mui/material/Chip";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import Stack from "@mui/material/Stack";
import LinearProgress from "@mui/material/LinearProgress";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import StopIcon from "@mui/icons-material/Stop";
import InfoIcon from "@mui/icons-material/Info";
import BarChartIcon from "@mui/icons-material/BarChart";
import CompareIcon from "@mui/icons-material/Compare";
import AssessmentIcon from "@mui/icons-material/Assessment";
import SettingsIcon from "@mui/icons-material/Settings";
import DownloadIcon from "@mui/icons-material/Download";
import ReplayIcon from "@mui/icons-material/Replay";
import VisibilityIcon from "@mui/icons-material/Visibility";

// 模拟数据 - 实际应用中从API获取
const mockAttackMethods = [
  { id: "basic", name: "基础反演攻击" },
  { id: "improved", name: "改进反演攻击" },
  { id: "gan", name: "GAN辅助反演" },
  { id: "gradient", name: "梯度泄露攻击" }
];

const mockTargetModels = [
  { id: "cnn", name: "简单CNN分类器" },
  { id: "resnet", name: "ResNet-18模型" },
  { id: "densenet", name: "DenseNet-121模型" }
];

const mockDatasets = [
  { id: "mnist", name: "MNIST", classes: 10 },
  { id: "cifar10", name: "CIFAR-10", classes: 10 },
  { id: "att_faces", name: "AT&T人脸数据库", classes: 40 }
];

// 模拟评估结果
const generateMockResults = () => {
  const methods = mockAttackMethods.map(m => m.id);
  const models = mockTargetModels.map(m => m.id);
  const results = [];

  for (const method of methods) {
    for (const model of models) {
      // 随机生成一些结果数据
      const accuracy = Math.random() * 0.6 + 0.2; // 20%-80%
      const successRate = Math.random() * 0.7 + 0.15; // 15%-85%
      const psnr = Math.random() * 15 + 10; // 10-25
      const ssim = Math.random() * 0.5 + 0.3; // 0.3-0.8
      const avgConfidence = Math.random() * 0.4 + 0.5; // 50%-90%
      const executionTime = Math.floor(Math.random() * 300 + 100); // 100-400秒
      
      // 类别敏感性 - 每个类别的成功率
      const classSuccessRates = Array.from({length: 10}, () => Math.random() * 0.8 + 0.1);
      
      // 置信度分布
      const confidenceDistribution = Array.from({length: 10}, (_, i) => ({
        confidenceRange: `${(0.1 * i).toFixed(1)}-${(0.1 * (i + 1)).toFixed(1)}`,
        count: Math.floor(Math.random() * 30)
      }));
      
      results.push({
        id: `eval-${method}-${model}-${Date.now()}`,
        method,
        model,
        dataset: "mnist", // 假设都是MNIST
        accuracy,
        successRate,
        psnr,
        ssim,
        avgConfidence,
        executionTime,
        timestamp: new Date().toISOString(),
        classSuccessRates,
        confidenceDistribution,
        sampleCount: 100,
        completedSamples: 100
      });
    }
  }
  
  return results;
};

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`evaluation-tabpanel-${index}`}
      aria-labelledby={`evaluation-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 2 }}>{children}</Box>}
    </div>
  );
}

// 结果表格组件
function ResultsTable({ results, onViewDetails }) {
  return (
    <TableContainer component={Paper} sx={{ mt: 2 }}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>攻击方法</TableCell>
            <TableCell>目标模型</TableCell>
            <TableCell>准确率</TableCell>
            <TableCell>攻击成功率</TableCell>
            <TableCell>PSNR</TableCell>
            <TableCell>SSIM</TableCell>
            <TableCell>平均置信度</TableCell>
            <TableCell>执行时间</TableCell>
            <TableCell align="center">操作</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {results.map((result) => {
            // 获取方法和模型的名称
            const methodName = mockAttackMethods.find(m => m.id === result.method)?.name || result.method;
            const modelName = mockTargetModels.find(m => m.id === result.model)?.name || result.model;
            
            return (
              <TableRow key={result.id} hover>
                <TableCell>{methodName}</TableCell>
                <TableCell>{modelName}</TableCell>
                <TableCell>{(result.accuracy * 100).toFixed(2)}%</TableCell>
                <TableCell>{(result.successRate * 100).toFixed(2)}%</TableCell>
                <TableCell>{result.psnr.toFixed(2)}</TableCell>
                <TableCell>{result.ssim.toFixed(2)}</TableCell>
                <TableCell>{(result.avgConfidence * 100).toFixed(2)}%</TableCell>
                <TableCell>{result.executionTime}秒</TableCell>
                <TableCell align="center">
                  <IconButton onClick={() => onViewDetails(result)} size="small" color="primary">
                    <VisibilityIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            );
          })}
          {results.length === 0 && (
            <TableRow>
              <TableCell colSpan={9} align="center">
                <Typography sx={{ py: 2 }}>暂无评估结果数据</Typography>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

// 比较图表组件
function ComparisonCharts({ selectedResults }) {
  if (selectedResults.length === 0) {
    return (
      <Alert severity="info" sx={{ mt: 2 }}>
        请先选择要比较的评估结果
      </Alert>
    );
  }

  // 以下是示例图表，实际应用中应使用Chart.js或Recharts等库
  // 这里使用简化的显示方式
  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        攻击效果比较
      </Typography>
      
      <Grid container spacing={3}>
        {/* 准确率比较 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1">准确率比较</Typography>
              <Box sx={{ mt: 2, height: 200, position: 'relative' }}>
                {selectedResults.map((result, index) => {
                  const methodName = mockAttackMethods.find(m => m.id === result.method)?.name || result.method;
                  const modelName = mockTargetModels.find(m => m.id === result.model)?.name || result.model;
                  const barHeight = 25;
                  const topOffset = index * (barHeight + 15) + 10;
                  
                  return (
                    <Box key={result.id} sx={{ position: 'absolute', top: topOffset, width: '100%' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        <Typography variant="body2" sx={{ width: 180, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {methodName} + {modelName}
                        </Typography>
                        <Box sx={{ flexGrow: 1, ml: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={result.accuracy * 100}
                            sx={{ 
                              height: barHeight, 
                              borderRadius: 1,
                              backgroundColor: 'rgba(0, 0, 0, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: `hsl(${index * 45}, 70%, 50%)`,
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ ml: 1, minWidth: 60 }}>
                          {(result.accuracy * 100).toFixed(2)}%
                        </Typography>
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* 攻击成功率比较 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1">攻击成功率比较</Typography>
              <Box sx={{ mt: 2, height: 200, position: 'relative' }}>
                {selectedResults.map((result, index) => {
                  const methodName = mockAttackMethods.find(m => m.id === result.method)?.name || result.method;
                  const modelName = mockTargetModels.find(m => m.id === result.model)?.name || result.model;
                  const barHeight = 25;
                  const topOffset = index * (barHeight + 15) + 10;
                  
                  return (
                    <Box key={result.id} sx={{ position: 'absolute', top: topOffset, width: '100%' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        <Typography variant="body2" sx={{ width: 180, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {methodName} + {modelName}
                        </Typography>
                        <Box sx={{ flexGrow: 1, ml: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={result.successRate * 100}
                            sx={{ 
                              height: barHeight, 
                              borderRadius: 1,
                              backgroundColor: 'rgba(0, 0, 0, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: `hsl(${index * 45 + 180}, 70%, 50%)`,
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ ml: 1, minWidth: 60 }}>
                          {(result.successRate * 100).toFixed(2)}%
                        </Typography>
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* 图像质量比较 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1">图像质量比较 (PSNR)</Typography>
              <Box sx={{ mt: 2, height: 200, position: 'relative' }}>
                {selectedResults.map((result, index) => {
                  const methodName = mockAttackMethods.find(m => m.id === result.method)?.name || result.method;
                  const modelName = mockTargetModels.find(m => m.id === result.model)?.name || result.model;
                  const barHeight = 25;
                  const topOffset = index * (barHeight + 15) + 10;
                  // PSNR 的范围通常是 0-50，但我们使用 0-30 作为最大范围
                  const normalizedValue = Math.min(100, (result.psnr / 30) * 100);
                  
                  return (
                    <Box key={result.id} sx={{ position: 'absolute', top: topOffset, width: '100%' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        <Typography variant="body2" sx={{ width: 180, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {methodName} + {modelName}
                        </Typography>
                        <Box sx={{ flexGrow: 1, ml: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={normalizedValue}
                            sx={{ 
                              height: barHeight, 
                              borderRadius: 1,
                              backgroundColor: 'rgba(0, 0, 0, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: `hsl(${index * 45 + 90}, 70%, 50%)`,
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ ml: 1, minWidth: 60 }}>
                          {result.psnr.toFixed(2)}
                        </Typography>
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* 执行时间比较 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1">执行时间比较 (秒)</Typography>
              <Box sx={{ mt: 2, height: 200, position: 'relative' }}>
                {selectedResults.map((result, index) => {
                  const methodName = mockAttackMethods.find(m => m.id === result.method)?.name || result.method;
                  const modelName = mockTargetModels.find(m => m.id === result.model)?.name || result.model;
                  const barHeight = 25;
                  const topOffset = index * (barHeight + 15) + 10;
                  // 假设最大执行时间为600秒
                  const normalizedValue = Math.min(100, (result.executionTime / 600) * 100);
                  
                  return (
                    <Box key={result.id} sx={{ position: 'absolute', top: topOffset, width: '100%' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        <Typography variant="body2" sx={{ width: 180, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {methodName} + {modelName}
                        </Typography>
                        <Box sx={{ flexGrow: 1, ml: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={normalizedValue}
                            sx={{ 
                              height: barHeight, 
                              borderRadius: 1,
                              backgroundColor: 'rgba(0, 0, 0, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: `hsl(${index * 45 + 270}, 70%, 50%)`,
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ ml: 1, minWidth: 60 }}>
                          {result.executionTime}
                        </Typography>
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        类别敏感性分析
      </Typography>
      
      <Grid container spacing={3}>
        {selectedResults.map((result, index) => {
          const methodName = mockAttackMethods.find(m => m.id === result.method)?.name || result.method;
          const modelName = mockTargetModels.find(m => m.id === result.model)?.name || result.model;
          
          return (
            <Grid item xs={12} md={6} key={`class-${result.id}`}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    {methodName} + {modelName} - 类别成功率
                  </Typography>
                  <Box sx={{ mt: 2, maxHeight: 300, overflowY: 'auto' }}>
                    {result.classSuccessRates.map((rate, classIndex) => (
                      <Box key={`class-${result.id}-${classIndex}`} sx={{ mb: 1.5 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                          <Typography variant="body2">
                            类别 {classIndex}
                          </Typography>
                          <Typography variant="body2">
                            {(rate * 100).toFixed(2)}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={rate * 100}
                          sx={{ 
                            height: 8, 
                            borderRadius: 1,
                            backgroundColor: 'rgba(0, 0, 0, 0.1)',
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: `hsl(${rate * 120}, 70%, 50%)`,
                            }
                          }}
                        />
                      </Box>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
      
      <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <Button variant="contained" startIcon={<DownloadIcon />}>
          导出对比报告
        </Button>
      </Box>
    </Box>
  );
}

// 详细结果组件
function ResultDetails({ result, onClose }) {
  if (!result) return null;
  
  const methodName = mockAttackMethods.find(m => m.id === result.method)?.name || result.method;
  const modelName = mockTargetModels.find(m => m.id === result.model)?.name || result.model;
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        评估详情: {methodName} + {modelName}
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                基本信息
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">攻击方法</Typography>
                  <Typography variant="body1">{methodName}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">目标模型</Typography>
                  <Typography variant="body1">{modelName}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">数据集</Typography>
                  <Typography variant="body1">
                    {mockDatasets.find(d => d.id === result.dataset)?.name || result.dataset}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">样本数量</Typography>
                  <Typography variant="body1">{result.sampleCount}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">评估时间</Typography>
                  <Typography variant="body1">
                    {new Date(result.timestamp).toLocaleString()}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">总执行时间</Typography>
                  <Typography variant="body1">{result.executionTime}秒</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                攻击效果
              </Typography>
              <Box sx={{ my: 1 }}>
                <Typography variant="body2" color="text.secondary">准确率</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                  <LinearProgress
                    variant="determinate"
                    value={result.accuracy * 100}
                    sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                  />
                  <Typography variant="body2" sx={{ ml: 2, minWidth: 50 }}>
                    {(result.accuracy * 100).toFixed(2)}%
                  </Typography>
                </Box>
              </Box>
              
              <Box sx={{ my: 1 }}>
                <Typography variant="body2" color="text.secondary">攻击成功率</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                  <LinearProgress
                    variant="determinate"
                    value={result.successRate * 100}
                    sx={{ 
                      flexGrow: 1, 
                      height: 8, 
                      borderRadius: 1,
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#4caf50',
                      }
                    }}
                  />
                  <Typography variant="body2" sx={{ ml: 2, minWidth: 50 }}>
                    {(result.successRate * 100).toFixed(2)}%
                  </Typography>
                </Box>
              </Box>
              
              <Box sx={{ my: 1 }}>
                <Typography variant="body2" color="text.secondary">平均置信度</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                  <LinearProgress
                    variant="determinate"
                    value={result.avgConfidence * 100}
                    sx={{ 
                      flexGrow: 1, 
                      height: 8, 
                      borderRadius: 1,
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#ff9800',
                      }
                    }}
                  />
                  <Typography variant="body2" sx={{ ml: 2, minWidth: 50 }}>
                    {(result.avgConfidence * 100).toFixed(2)}%
                  </Typography>
                </Box>
              </Box>
              
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">PSNR</Typography>
                  <Typography variant="body1">{result.psnr.toFixed(2)} dB</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">SSIM</Typography>
                  <Typography variant="body1">{result.ssim.toFixed(3)}</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                置信度分布
              </Typography>
              <Box sx={{ mt: 2, height: 200, position: 'relative' }}>
                {result.confidenceDistribution.map((item, index) => {
                  const barHeight = 15;
                  const topOffset = index * (barHeight + 10) + 5;
                  // 找出最大值用于归一化
                  const maxCount = Math.max(...result.confidenceDistribution.map(item => item.count));
                  const normalizedValue = Math.min(100, (item.count / maxCount) * 100);
                  
                  return (
                    <Box key={`conf-${index}`} sx={{ position: 'absolute', top: topOffset, width: '100%' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        <Typography variant="body2" sx={{ width: 100 }}>
                          {item.confidenceRange}
                        </Typography>
                        <Box sx={{ flexGrow: 1, ml: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={normalizedValue}
                            sx={{ 
                              height: barHeight, 
                              borderRadius: 1,
                              backgroundColor: 'rgba(0, 0, 0, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: `rgba(33, 150, 243, ${0.3 + index * 0.07})`,
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ ml: 1, minWidth: 30 }}>
                          {item.count}
                        </Typography>
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                类别敏感性
              </Typography>
              <Box sx={{ mt: 2, maxHeight: 265, overflowY: 'auto' }}>
                {result.classSuccessRates.map((rate, classIndex) => (
                  <Box key={`detail-class-${classIndex}`} sx={{ mb: 1.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2">
                        类别 {classIndex}
                      </Typography>
                      <Typography variant="body2">
                        {(rate * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={rate * 100}
                      sx={{ 
                        height: 8, 
                        borderRadius: 1,
                        backgroundColor: 'rgba(0, 0, 0, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: `hsl(${rate * 120}, 70%, 50%)`,
                        }
                      }}
                    />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
        <Button variant="outlined" sx={{ mr: 2 }} onClick={onClose}>
          返回
        </Button>
        <Button variant="contained" startIcon={<DownloadIcon />}>
          导出详细报告
        </Button>
      </Box>
    </Box>
  );
}

// 批量攻击配置表单
function BatchAttackForm({ onStart, isRunning }) {
  const [dataset, setDataset] = useState("mnist");
  const [targetModel, setTargetModel] = useState("cnn");
  const [attackMethods, setAttackMethods] = useState(["basic"]);
  const [labelRange, setLabelRange] = useState({ start: 0, end: 9 });
  const [samplesPerLabel, setSamplesPerLabel] = useState(10);
  const [advancedSettings, setAdvancedSettings] = useState({
    iterations: 1000,
    learningRate: 0.01,
    useRegularization: true,
    batchSize: 32
  });

  const handleAttackMethodToggle = (method) => {
    setAttackMethods(prev => 
      prev.includes(method)
        ? prev.filter(m => m !== method)
        : [...prev, method]
    );
  };

  const handleStartEvaluation = () => {
    // 准备评估配置
    const config = {
      dataset,
      targetModel,
      attackMethods,
      labelRange,
      samplesPerLabel,
      advancedSettings
    };
    
    // 调用父组件的开始函数
    onStart(config);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        配置批量攻击评估
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>数据集</InputLabel>
            <Select
              value={dataset}
              label="数据集"
              onChange={(e) => setDataset(e.target.value)}
              disabled={isRunning}
            >
              {mockDatasets.map((dataset) => (
                <MenuItem key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.classes}类)
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>目标模型</InputLabel>
            <Select
              value={targetModel}
              label="目标模型"
              onChange={(e) => setTargetModel(e.target.value)}
              disabled={isRunning}
            >
              {mockTargetModels.map((model) => (
                <MenuItem key={model.id} value={model.id}>
                  {model.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              攻击方法（可多选）
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {mockAttackMethods.map((method) => (
                <Chip
                  key={method.id}
                  label={method.name}
                  clickable
                  color={attackMethods.includes(method.id) ? "primary" : "default"}
                  onClick={() => handleAttackMethodToggle(method.id)}
                  disabled={isRunning}
                />
              ))}
            </Box>
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>
            标签范围
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <TextField
              label="起始标签"
              type="number"
              value={labelRange.start}
              onChange={(e) => setLabelRange({...labelRange, start: parseInt(e.target.value) || 0})}
              InputProps={{ inputProps: { min: 0 } }}
              disabled={isRunning}
              size="small"
              fullWidth
            />
            <TextField
              label="结束标签"
              type="number"
              value={labelRange.end}
              onChange={(e) => setLabelRange({...labelRange, end: parseInt(e.target.value) || 0})}
              InputProps={{ inputProps: { min: labelRange.start } }}
              disabled={isRunning}
              size="small"
              fullWidth
            />
          </Box>
          
          <TextField
            fullWidth
            label="每个标签的样本数"
            type="number"
            value={samplesPerLabel}
            onChange={(e) => setSamplesPerLabel(parseInt(e.target.value) || 1)}
            InputProps={{ inputProps: { min: 1 } }}
            disabled={isRunning}
            sx={{ mb: 2 }}
          />
          
          <Accordion disabled={isRunning}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>高级参数设置</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="迭代次数"
                    type="number"
                    value={advancedSettings.iterations}
                    onChange={(e) => setAdvancedSettings({...advancedSettings, iterations: parseInt(e.target.value) || 100})}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="学习率"
                    type="number"
                    value={advancedSettings.learningRate}
                    onChange={(e) => setAdvancedSettings({...advancedSettings, learningRate: parseFloat(e.target.value) || 0.01})}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="批次大小"
                    type="number"
                    value={advancedSettings.batchSize}
                    onChange={(e) => setAdvancedSettings({...advancedSettings, batchSize: parseInt(e.target.value) || 32})}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={advancedSettings.useRegularization}
                        onChange={(e) => setAdvancedSettings({...advancedSettings, useRegularization: e.target.checked})}
                      />
                    }
                    label="使用正则化"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>
      
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          color="primary"
          startIcon={isRunning ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
          onClick={handleStartEvaluation}
          disabled={isRunning || attackMethods.length === 0}
          sx={{ minWidth: 200 }}
        >
          {isRunning ? "评估中..." : "开始评估"}
        </Button>
      </Box>
      
      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2">
          将执行 {attackMethods.length} 种攻击方法 x {labelRange.end - labelRange.start + 1} 个标签 x {samplesPerLabel} 个样本 = 
          <strong> {attackMethods.length * (labelRange.end - labelRange.start + 1) * samplesPerLabel}</strong> 次攻击
        </Typography>
      </Alert>
    </Box>
  );
}

// 评估任务进度组件
function EvaluationProgress({ task, onStop }) {
  if (!task) return null;
  
  const totalSamples = task.totalSamples || 100;
  const completedSamples = task.completedSamples || 0;
  const progress = Math.round((completedSamples / totalSamples) * 100);
  
  // 计算预计剩余时间
  const elapsedTime = task.elapsedTime || 0; // 单位：秒
  const remainingTime = completedSamples > 0 
    ? Math.round((elapsedTime / completedSamples) * (totalSamples - completedSamples))
    : 0;
  
  // 格式化时间
  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    return `${hours > 0 ? `${hours}小时 ` : ''}${minutes > 0 ? `${minutes}分钟 ` : ''}${secs}秒`;
  };
  
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        评估任务进度
      </Typography>
      
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={8}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Box sx={{ width: '100%', mr: 1 }}>
                  <LinearProgress variant="determinate" value={progress} sx={{ height: 10, borderRadius: 5 }} />
                </Box>
                <Box sx={{ minWidth: 50 }}>
                  <Typography variant="body2" color="text.secondary">{progress}%</Typography>
                </Box>
              </Box>
              
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                已完成：{completedSamples} / {totalSamples} 样本
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Typography variant="body2">
                耗时：{formatTime(elapsedTime)}
              </Typography>
              {remainingTime > 0 && (
                <Typography variant="body2">
                  预计剩余：{formatTime(remainingTime)}
                </Typography>
              )}
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 2 }} />
          
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="subtitle2">当前数据集</Typography>
              <Typography variant="body2">
                {mockDatasets.find(d => d.id === task.dataset)?.name || task.dataset}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="subtitle2">目标模型</Typography>
              <Typography variant="body2">
                {mockTargetModels.find(m => m.id === task.targetModel)?.name || task.targetModel}
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="subtitle2">攻击方法</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                {task.attackMethods.map(methodId => (
                  <Chip 
                    key={methodId}
                    label={mockAttackMethods.find(m => m.id === methodId)?.name || methodId}
                    size="small"
                  />
                ))}
              </Box>
            </Grid>
          </Grid>
          
          {task.currentLabel !== undefined && task.currentMethod && (
            <Alert severity="info" sx={{ mt: 2 }}>
              当前正在处理：
              <strong> {mockAttackMethods.find(m => m.id === task.currentMethod)?.name || task.currentMethod} </strong>
              攻击，目标标签：
              <strong> {task.currentLabel} </strong>
            </Alert>
          )}
        </CardContent>
      </Card>
      
      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          color="error"
          startIcon={<StopIcon />}
          onClick={onStop}
        >
          停止评估
        </Button>
      </Box>
    </Box>
  );
}

export default function AttackEvaluationPage() {
  const [pageTitle, setPageTitle] = useState("攻击效果评估");
  const [activeTab, setActiveTab] = useState(0);
  const [evaluationResults, setEvaluationResults] = useState([]);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [currentTask, setCurrentTask] = useState(null);
  const [selectedResultIds, setSelectedResultIds] = useState([]);
  const [viewingResult, setViewingResult] = useState(null);
  
  // 加载评估结果
  useEffect(() => {
    // 模拟从API加载
    // 实际应用中应调用后端API
    setEvaluationResults(generateMockResults());
  }, []);
  
  // 开始评估
  const handleStartEvaluation = (config) => {
    setIsEvaluating(true);
    
    // 创建评估任务
    const newTask = {
      id: `task-${Date.now()}`,
      dataset: config.dataset,
      targetModel: config.targetModel,
      attackMethods: config.attackMethods,
      labelRange: config.labelRange,
      samplesPerLabel: config.samplesPerLabel,
      advancedSettings: config.advancedSettings,
      totalSamples: config.attackMethods.length * (config.labelRange.end - config.labelRange.start + 1) * config.samplesPerLabel,
      completedSamples: 0,
      elapsedTime: 0,
      currentLabel: config.labelRange.start,
      currentMethod: config.attackMethods[0],
      startTime: Date.now()
    };
    
    setCurrentTask(newTask);
    
    // 模拟评估进度
    // 实际应用中应调用后端API并接收进度更新
    const progressInterval = setInterval(() => {
      setCurrentTask(prev => {
        // 计算新的已完成样本数
        let newCompleted = prev.completedSamples + Math.floor(Math.random() * 3) + 1;
        newCompleted = Math.min(newCompleted, prev.totalSamples);
        
        // 计算经过的时间
        const elapsedTime = Math.floor((Date.now() - prev.startTime) / 1000);
        
        // 更新当前处理的标签和方法
        let currentLabel = prev.currentLabel;
        let currentMethod = prev.currentMethod;
        
        // 根据进度更新当前标签和方法
        const methodIndex = prev.attackMethods.indexOf(prev.currentMethod);
        const labelsPerMethod = (prev.labelRange.end - prev.labelRange.start + 1);
        const samplesPerMethod = labelsPerMethod * prev.samplesPerLabel;
        const completedMethods = Math.floor(newCompleted / samplesPerMethod);
        const remainingSamples = newCompleted % samplesPerMethod;
        const completedLabelsInMethod = Math.floor(remainingSamples / prev.samplesPerLabel);
        
        if (completedMethods < prev.attackMethods.length) {
          currentMethod = prev.attackMethods[completedMethods];
          currentLabel = prev.labelRange.start + completedLabelsInMethod;
        }
        
        // 检查是否完成
        if (newCompleted >= prev.totalSamples) {
          clearInterval(progressInterval);
          
          // 模拟生成最终结果
          setTimeout(() => {
            const newResults = [];
            // 为每种攻击方法生成一个结果
            for (const method of prev.attackMethods) {
              newResults.push({
                id: `eval-${method}-${prev.targetModel}-${Date.now()}`,
                method,
                model: prev.targetModel,
                dataset: prev.dataset,
                accuracy: Math.random() * 0.6 + 0.2,
                successRate: Math.random() * 0.7 + 0.15,
                psnr: Math.random() * 15 + 10,
                ssim: Math.random() * 0.5 + 0.3,
                avgConfidence: Math.random() * 0.4 + 0.5,
                executionTime: elapsedTime / prev.attackMethods.length,
                timestamp: new Date().toISOString(),
                classSuccessRates: Array.from(
                  {length: mockDatasets.find(d => d.id === prev.dataset)?.classes || 10},
                  () => Math.random() * 0.8 + 0.1
                ),
                confidenceDistribution: Array.from({length: 10}, (_, i) => ({
                  confidenceRange: `${(0.1 * i).toFixed(1)}-${(0.1 * (i + 1)).toFixed(1)}`,
                  count: Math.floor(Math.random() * 30)
                })),
                sampleCount: prev.totalSamples / prev.attackMethods.length,
                completedSamples: prev.totalSamples / prev.attackMethods.length
              });
            }
            
            // 添加到结果集
            setEvaluationResults(prevResults => [...newResults, ...prevResults]);
            setIsEvaluating(false);
            setCurrentTask(null);
            
            // 切换到结果标签
            setActiveTab(1);
          }, 1000);
        }
        
        return {
          ...prev,
          completedSamples: newCompleted,
          elapsedTime,
          currentLabel,
          currentMethod
        };
      });
    }, 500);
    
    // 切换到进度标签
    setActiveTab(2);
  };
  
  // 停止评估
  const handleStopEvaluation = () => {
    // 实际应用中应调用后端API停止任务
    setIsEvaluating(false);
    setCurrentTask(null);
  };
  
  // 查看详细结果
  const handleViewDetails = (result) => {
    setViewingResult(result);
    setActiveTab(3);
  };
  
  // 返回结果列表
  const handleBackToResults = () => {
    setViewingResult(null);
    setActiveTab(1);
  };
  
  // 选择结果进行比较
  const handleSelectForComparison = (resultId) => {
    setSelectedResultIds(prev => 
      prev.includes(resultId)
        ? prev.filter(id => id !== resultId)
        : [...prev, resultId]
    );
  };
  
  // 筛选选中的结果
  const selectedResults = evaluationResults.filter(result => 
    selectedResultIds.includes(result.id)
  );
  
  return (
    <div>
      <h1>{pageTitle}</h1>
      <div style={{ marginTop: "30px", marginBottom: "30px" }}></div>
      <Box sx={{ width: '100%', typography: 'body1' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={activeTab} 
            onChange={(e, newValue) => setActiveTab(newValue)}
            aria-label="attack evaluation tabs"
          >
            <Tab 
              label="配置评估" 
              icon={<SettingsIcon />} 
              iconPosition="start" 
            />
            <Tab 
              label="评估结果" 
              icon={<AssessmentIcon />} 
              iconPosition="start" 
            />
            <Tab 
              label="进度监控" 
              icon={<BarChartIcon />} 
              iconPosition="start" 
              disabled={!isEvaluating}
            />
            <Tab 
              label="详细结果" 
              icon={<InfoIcon />} 
              iconPosition="start" 
              disabled={!viewingResult}
            />
            <Tab 
              label="对比分析" 
              icon={<CompareIcon />} 
              iconPosition="start" 
            />
          </Tabs>
        </Box>
        
        <TabPanel value={activeTab} index={0}>
          <BatchAttackForm onStart={handleStartEvaluation} isRunning={isEvaluating} />
        </TabPanel>
        
        <TabPanel value={activeTab} index={1}>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              评估结果
            </Typography>
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                共 {evaluationResults.length} 条结果
              </Typography>
              
              <Box>
                <Button 
                  variant="outlined" 
                  startIcon={<CompareIcon />}
                  onClick={() => setActiveTab(4)}
                  disabled={selectedResultIds.length < 2}
                  sx={{ mr: 1 }}
                >
                  对比选中的 ({selectedResultIds.length})
                </Button>
                
                <Button 
                  variant="outlined" 
                  startIcon={<ReplayIcon />}
                  onClick={() => setEvaluationResults(generateMockResults())}
                >
                  刷新
                </Button>
              </Box>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                选择结果进行比较
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {evaluationResults.map((result) => {
                  const methodName = mockAttackMethods.find(m => m.id === result.method)?.name || result.method;
                  const modelName = mockTargetModels.find(m => m.id === result.model)?.name || result.model;
                  const isSelected = selectedResultIds.includes(result.id);
                  
                  return (
                    <Chip
                      key={result.id}
                      label={`${methodName} + ${modelName}`}
                      clickable
                      color={isSelected ? "primary" : "default"}
                      onClick={() => handleSelectForComparison(result.id)}
                      sx={{ mb: 1 }}
                    />
                  );
                })}
              </Box>
            </Box>
          </Box>
          
          <ResultsTable 
            results={evaluationResults} 
            onViewDetails={handleViewDetails} 
          />
        </TabPanel>
        
        <TabPanel value={activeTab} index={2}>
          <EvaluationProgress task={currentTask} onStop={handleStopEvaluation} />
        </TabPanel>
        
        <TabPanel value={activeTab} index={3}>
          <ResultDetails result={viewingResult} onClose={handleBackToResults} />
        </TabPanel>
        
        <TabPanel value={activeTab} index={4}>
          <Typography variant="h6" gutterBottom>
            对比分析
          </Typography>
          
          <ComparisonCharts selectedResults={selectedResults} />
        </TabPanel>
      </Box>
    </div>
  );
}// 评估任务进度组件