package gateway_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ervijay/polysent/internal/engine"
	"github.com/ervijay/polysent/internal/engine/tier1"
	"github.com/ervijay/polysent/internal/gateway"
	"github.com/ervijay/polysent/internal/router"
)

func setupTestServer(t *testing.T) *gateway.Server {
	weightsPath := filepath.Join("..", "..", "models", "tier1_tfidf", "weights.json")
	t1, err := tier1.NewTFIDFAnalyzer(weightsPath)
	if err != nil {
		t.Fatalf("Failed to initialize Tier-1: %v", err)
	}

	r := router.NewRouter(t1, nil, nil)
	return gateway.NewServer(r)
}

func TestHealthEndpoint(t *testing.T) {
	srv := setupTestServer(t)
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if resp["status"] != "ok" {
		t.Errorf("Expected status 'ok', got %v", resp["status"])
	}
}

func TestSwaggerEndpoints(t *testing.T) {
	srv := setupTestServer(t)

	// 1. Raw OpenAPI spec
	reqSpec := httptest.NewRequest("GET", "/swagger/openapi.yaml", nil)
	wSpec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(wSpec, reqSpec)

	if wSpec.Code != http.StatusOK {
		t.Errorf("Expected status 200 for openapi.yaml, got %d", wSpec.Code)
	}
	if !strings.Contains(wSpec.Body.String(), "openapi: 3.0.3") {
		t.Errorf("Expected OpenAPI 3.0.3 content in spec response")
	}

	// 2. Swagger UI HTML
	reqUI := httptest.NewRequest("GET", "/swagger/", nil)
	wUI := httptest.NewRecorder()
	srv.Handler().ServeHTTP(wUI, reqUI)

	if wUI.Code != http.StatusOK {
		t.Errorf("Expected status 200 for /swagger/, got %d", wUI.Code)
	}
	if !strings.Contains(wUI.Body.String(), "SwaggerUIBundle") {
		t.Errorf("Expected SwaggerUIBundle in HTML response")
	}

	// 3. /docs redirect
	reqDocs := httptest.NewRequest("GET", "/docs", nil)
	wDocs := httptest.NewRecorder()
	srv.Handler().ServeHTTP(wDocs, reqDocs)

	if wDocs.Code != http.StatusMovedPermanently {
		t.Errorf("Expected status 301 for /docs, got %d", wDocs.Code)
	}
}

func TestAnalyzeEndpoint(t *testing.T) {
	srv := setupTestServer(t)

	// 1. Valid request
	body, _ := json.Marshal(map[string]interface{}{
		"text":     "The display is crisp and vivid, truly amazing!",
		"task":     "document_level",
		"strategy": "ultra_fast",
	})
	req := httptest.NewRequest("POST", "/v1/sentiment/analyze", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp engine.Response
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if resp.Status != "success" {
		t.Errorf("Expected status 'success', got %s", resp.Status)
	}
	if resp.Overall == nil || resp.Overall.Label != "positive" {
		t.Errorf("Expected label 'positive', got %v", resp.Overall)
	}

	// 2. Empty text request -> 400
	badBody, _ := json.Marshal(map[string]interface{}{
		"text": "",
	})
	badReq := httptest.NewRequest("POST", "/v1/sentiment/analyze", bytes.NewReader(badBody))
	badW := httptest.NewRecorder()
	srv.Handler().ServeHTTP(badW, badReq)

	if badW.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for empty text, got %d", badW.Code)
	}
}
