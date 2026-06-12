FROM nginx:1.27-alpine

COPY visualizations/ /usr/share/nginx/html/

EXPOSE 80
