FROM hugomods/hugo:std-exts-0.132.2 AS builder

WORKDIR /site
COPY . /site
RUN hugo

FROM nginx:alpine
COPY --from=builder /site/public /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]