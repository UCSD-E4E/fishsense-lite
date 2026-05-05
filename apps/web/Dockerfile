# fishsense web (Next.js App Router landing page)
#
# IMPORTANT: build with the **repo root** as context, matching the
# convention used by the Python services. Currently the web app has no
# JS workspace members to pull in, but anchoring at repo root means a
# future shared-TS lib under libs/ would not require changing this
# build invocation.
#
#   docker build -t fishsense-web \
#     -f apps/web/Dockerfile .

FROM node:24-alpine AS deps
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm install --no-audit --no-fund --loglevel=error

FROM node:24-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY apps/web/ ./
RUN npm run build

FROM node:24-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup -g 1001 -S nodejs && adduser -S -u 1001 -G nodejs nextjs
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
USER nextjs
EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0
CMD ["node", "server.js"]
