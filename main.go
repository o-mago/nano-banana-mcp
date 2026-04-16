// nano-banana MCP server
//
// Exposes two tools via the Model Context Protocol (MCP):
//   - generate_image: generates a single image from a text prompt
//   - generate_lip_sync_images: generates mouth-open + mouth-closed image pair
//
// Uses Google Gemini for image generation and serves over SSE transport.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"

	"github.com/google/generative-ai-go/genai"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/api/option"
)

const imageModel = "gemini-2.5-flash-preview"

type GenerateImageArgs struct {
	Prompt string `json:"prompt"`
}

type GenerateLipSyncArgs struct {
	Prompt string `json:"prompt"`
}

func expandHome(path string) string {
	if len(path) >= 2 && path[:2] == "~/" {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return filepath.Join(home, path[2:])
	}
	return path
}

func newGeminiClient(ctx context.Context) (*genai.Client, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GOOGLE_API_KEY environment variable not set")
	}
	return genai.NewClient(ctx, option.WithAPIKey(apiKey))
}

func ensureImageDir() (string, error) {
	dir := expandHome("~/image_gen")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("failed to create image directory: %w", err)
	}
	return dir, nil
}

func extractImageData(resp *genai.GenerateContentResponse) ([]byte, error) {
	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates in response")
	}
	candidate := resp.Candidates[0]
	if candidate.Content != nil {
		for _, part := range candidate.Content.Parts {
			if blob, ok := part.(genai.Blob); ok {
				return blob.Data, nil
			}
		}
	}
	msg := "no image data found in response"
	if candidate.FinishReason != 0 {
		msg += fmt.Sprintf(". Finish reason: %v", candidate.FinishReason)
	}
	return nil, fmt.Errorf("%s", msg)
}

func textResult(data any) (*mcp.CallToolResultFor[any], error) {
	b, _ := json.Marshal(data)
	return &mcp.CallToolResultFor[any]{
		Content:           []mcp.Content{&mcp.TextContent{Text: string(b)}},
		StructuredContent: data,
	}, nil
}

func generateImageHandler(ctx context.Context, _ *mcp.ServerSession, params *mcp.CallToolParamsFor[GenerateImageArgs]) (*mcp.CallToolResultFor[any], error) {
	dir, err := ensureImageDir()
	if err != nil {
		return nil, err
	}

	client, err := newGeminiClient(ctx)
	if err != nil {
		return nil, err
	}
	defer client.Close()

	model := client.GenerativeModel(imageModel)
	resp, err := model.GenerateContent(ctx, genai.Text(params.Arguments.Prompt))
	if err != nil {
		return nil, err
	}

	data, err := extractImageData(resp)
	if err != nil {
		return nil, err
	}

	imagePath := filepath.Join(dir, "char.png")
	if err := os.WriteFile(imagePath, data, 0644); err != nil {
		return nil, fmt.Errorf("failed to save image: %w", err)
	}

	return textResult(map[string]string{"status": "success", "image_path": imagePath})
}

func generateLipSyncImagesHandler(ctx context.Context, _ *mcp.ServerSession, params *mcp.CallToolParamsFor[GenerateLipSyncArgs]) (*mcp.CallToolResultFor[any], error) {
	dir, err := ensureImageDir()
	if err != nil {
		return nil, err
	}

	client, err := newGeminiClient(ctx)
	if err != nil {
		return nil, err
	}
	defer client.Close()

	model := client.GenerativeModel(imageModel)

	// Generate mouth-open image
	respOpen, err := model.GenerateContent(ctx, genai.Text(params.Arguments.Prompt+" with mouth open"))
	if err != nil {
		return nil, err
	}

	openData, err := extractImageData(respOpen)
	if err != nil {
		return nil, fmt.Errorf("mouth open image: %w", err)
	}

	openImagePath := filepath.Join(dir, "char-mouth-open.png")
	if err := os.WriteFile(openImagePath, openData, 0644); err != nil {
		return nil, fmt.Errorf("failed to save mouth-open image: %w", err)
	}

	// Generate mouth-closed image using the open image as reference
	respClosed, err := model.GenerateContent(ctx,
		genai.Text("change the mouth from open to close"),
		genai.ImageData("png", openData),
	)
	if err != nil {
		return nil, err
	}

	closedData, err := extractImageData(respClosed)
	if err != nil {
		return nil, fmt.Errorf("mouth closed image: %w", err)
	}

	closedImagePath := filepath.Join(dir, "char-mouth-closed.png")
	if err := os.WriteFile(closedImagePath, closedData, 0644); err != nil {
		return nil, fmt.Errorf("failed to save mouth-closed image: %w", err)
	}

	return textResult(map[string]string{
		"status":            "success",
		"open_image_path":   openImagePath,
		"closed_image_path": closedImagePath,
	})
}

func main() {
	s := mcp.NewServer(&mcp.Implementation{
		Name:    "nano_banana",
		Version: "1.0.0",
	}, nil)

	mcp.AddTool(s, &mcp.Tool{
		Name:        "generate_image",
		Description: "Generates an image based on a textual prompt using a generative model.",
	}, generateImageHandler)

	mcp.AddTool(s, &mcp.Tool{
		Name:        "generate_lip_sync_images",
		Description: "Generates two images for a lip-syncing app, one with mouth open and one with mouth closed.",
	}, generateLipSyncImagesHandler)

	fmt.Fprintln(os.Stderr, "Starting MCP server with SSE transport on :8080...")
	handler := mcp.NewSSEHandler(func(r *http.Request) *mcp.Server { return s })
	http.Handle("/", handler)
	// http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
	// 	fmt.Fprintf(os.Stderr, "Request: %s %s\n", r.Method, r.URL)
	// 	handler.ServeHTTP(w, r)
	// })
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}
