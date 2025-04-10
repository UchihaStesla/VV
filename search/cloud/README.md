# 语义搜索 (Cloudflare Worker)

基于Cloudflare Workers构建的语义搜索服务

## API使用说明

### 请求方式
```http
GET /?query={搜索文本}[&min_similarity={最小相似度}][&max_results={最大结果数}][&min_ratio={最小图片相似度}][&debug=true]
```

### 参数说明

| 参数 | 类型 | 必填 | 说明 | 默认值 | 取值范围 |
|------|------|------|------|------|------|
| query | string | 是 | 搜索文本 | - | - |
| min_similarity | float | 否 | 文本最小相似度阈值 | 0.5 | 0-1 |
| max_results | int | 否 | 返回最大结果数 | 10 | 1-20 |
| min_ratio | float | 否 | 图片相似度阈值 | 无 | 0-100 |
| debug | boolean | 否 | 启用调试模式 | false | true/false |

> **生产环境调试说明**：
> 1. 修改[worker.ts](./worker.ts)文件中的`ALLOW_DEBUG`变量为`true`
> 2. 在请求URL后添加`debug=true`参数
> 
> 注意：调试模式会返回敏感信息，生产环境请谨慎使用

### 成功响应
```json
[
  {
    "text": "字幕文本1",
    "timestamp": "时间戳1",
    "filename": "文件名1", 
    "text_similarity": 0.85,
    "image_similarity": 0.75
  },
  {
    "text": "字幕文本2",
    "timestamp": "时间戳2",
    "filename": "文件名2",
    "text_similarity": 0.82,
    "image_similarity": 0.68
  }
]
```

### 错误响应
```json
{
  "error": "错误信息",
  "debug": ["调试日志"] // 仅debug模式
}
```

## Cloudflare Token

在[Cloudflare Dashboard Token创建索引](https://dash.cloudflare.com/profile/api-tokens)需要配置以下权限：

**Workers相关权限**:
- Workers 脚本: 编辑
- Workers 构建配置: 编辑 
- Workers 管道: 编辑

**AI服务权限**:
- Workers AI: 编辑（如果使用Silicon Flow API则不需要）
- Vectorize: 编辑

**其他权限**:
- 帐户设置: 读取

## 构建云端向量索引

### 环境变量配置

| 变量名 | 说明 | 示例值 |
|------|------|------|
| SF_API_KEY | Silicon Flow API密钥 | sk-xxx |
| CLOUDFLARE_API_TOKEN | Cloudflare API Token | xxx |
| CLOUDFLARE_ACCOUNT_ID | Cloudflare账户ID | xxx |

> **API切换说明**：
> 1. 如使用硅基流动的`BAAI/bge-m3`模型：
>    - 配置SF_API_KEY环境变量（默认）
>    - 在[index-data.ts](./tools/index-data.ts#L18)中修改`USE_SF_API`变量为true
> 2. 否则使用Cloudflare Worker AI的`@cf/baai/bge-m3`模型

### 1. 创建索引
```bash
npm run index
```

### 2. 调试模式索引
```bash
npm run index:debug
```

### 3. 删除索引
```bash
npm run index:delete
```

## Worker部署

### 配置步骤

1. 执行部署命令：
```bash
npm run deploy
```

2. 部署完成后，在[Cloudflare Dashboard](https://dash.cloudflare.com/)中：
   - 进入Workers服务
   - 找到已部署的`subtitle-search` Worker

3. 在"变量和机密"：
   - 点击"添加变量"
   - 类型选择"密钥"
   - 名称填写: `SF_API_KEY` 
   - 值填入: 你的Silicon Flow API密钥

> **自动切换说明**：
> - Worker会优先检查`SF_API_KEY` Secret
> - 如果未设置该Secret，将自动使用Cloudflare Worker AI的API
