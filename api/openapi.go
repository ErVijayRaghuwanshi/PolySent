package api

import (
	_ "embed"
)

// OpenAPISpec contains the embedded raw OpenAPI 3.0 YAML specification.
//
//go:embed openapi.yaml
var OpenAPISpec []byte
