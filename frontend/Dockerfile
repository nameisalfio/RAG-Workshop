# Image of the web interface: static files served by nginx, which also forwards /api to the backend.
FROM nginx:1.29-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html styles.css app.js chats.js render.js evaluation.js /usr/share/nginx/html/
