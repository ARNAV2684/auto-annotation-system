# ──────────────────────────────────────────────────────────────────────────
# Frontend image, two stages:
#   1. node builds the static bundle (TypeScript check + Vite build)
#   2. nginx serves it and proxies /api/* to the backend container
# The final image contains only nginx + static files (~25 MB) — no Node.
# ──────────────────────────────────────────────────────────────────────────
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
