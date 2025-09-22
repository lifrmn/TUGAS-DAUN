// 🔧 SERVICE WORKER - PLANT CLASSIFIER PWA
// =========================================

const CACHE_NAME = 'plant-classifier-v1.0.0';
const STATIC_CACHE = 'static-cache-v1';
const DYNAMIC_CACHE = 'dynamic-cache-v1';

// 📦 Files to cache for offline use
const STATIC_FILES = [
    './',
    './index-super.html',
    './knn-super.js',
    './admin.html',
    './admin-dashboard.js',
    './manifest.json',
    'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
    'https://cdn.jsdelivr.net/npm/chart.js',
    'https://cdnjs.cloudflare.com/ajax/libs/particles.js/2.0.0/particles.min.js'
];

// 🚀 Install Event - Cache static files
self.addEventListener('install', event => {
    console.log('🔧 Service Worker: Installing...');
    
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then(cache => {
                console.log('📦 Caching static files...');
                return cache.addAll(STATIC_FILES);
            })
            .then(() => {
                console.log('✅ Static files cached successfully!');
                self.skipWaiting();
            })
            .catch(error => {
                console.error('❌ Error caching static files:', error);
            })
    );
});

// 🔄 Activate Event - Clean old caches
self.addEventListener('activate', event => {
    console.log('🔧 Service Worker: Activating...');
    
    event.waitUntil(
        caches.keys()
            .then(cacheNames => {
                return Promise.all(
                    cacheNames.map(cache => {
                        if (cache !== STATIC_CACHE && cache !== DYNAMIC_CACHE) {
                            console.log('🗑️ Deleting old cache:', cache);
                            return caches.delete(cache);
                        }
                    })
                );
            })
            .then(() => {
                console.log('✅ Service Worker activated!');
                self.clients.claim();
            })
    );
});

// 🌐 Fetch Event - Serve cached content
self.addEventListener('fetch', event => {
    const { request } = event;
    
    // Skip non-GET requests
    if (request.method !== 'GET') return;
    
    // Skip Chrome extension requests
    if (request.url.startsWith('chrome-extension://')) return;
    
    event.respondWith(
        caches.match(request)
            .then(cachedResponse => {
                if (cachedResponse) {
                    console.log('📋 Serving from cache:', request.url);
                    return cachedResponse;
                }
                
                // Fetch from network and cache dynamically
                return fetch(request)
                    .then(networkResponse => {
                        // Only cache successful responses
                        if (networkResponse.status === 200) {
                            const responseClone = networkResponse.clone();
                            
                            caches.open(DYNAMIC_CACHE)
                                .then(cache => {
                                    cache.put(request, responseClone);
                                    console.log('💾 Cached dynamically:', request.url);
                                });
                        }
                        
                        return networkResponse;
                    })
                    .catch(error => {
                        console.error('❌ Fetch failed:', error);
                        
                        // Return offline fallback for HTML pages
                        if (request.headers.get('accept').includes('text/html')) {
                            return caches.match('./index-super.html');
                        }
                        
                        // Return offline message for other requests
                        return new Response(
                            JSON.stringify({
                                error: 'Offline',
                                message: 'You are currently offline. Please check your internet connection.'
                            }),
                            {
                                headers: { 'Content-Type': 'application/json' },
                                status: 503
                            }
                        );
                    });
            })
    );
});

// 🔔 Push Notification Event
self.addEventListener('push', event => {
    console.log('🔔 Push notification received:', event);
    
    const options = {
        body: event.data ? event.data.text() : 'New plant classification available!',
        icon: './manifest.json',
        badge: './manifest.json',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'Open App',
                icon: './manifest.json'
            },
            {
                action: 'close',
                title: 'Close',
                icon: './manifest.json'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('🌿 Plant Classifier Pro', options)
    );
});

// 🎯 Notification Click Event
self.addEventListener('notificationclick', event => {
    console.log('🎯 Notification clicked:', event);
    
    event.notification.close();
    
    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('./index-super.html')
        );
    }
});

// 📊 Background Sync Event
self.addEventListener('sync', event => {
    console.log('🔄 Background sync triggered:', event.tag);
    
    if (event.tag === 'background-sync') {
        event.waitUntil(doBackgroundSync());
    }
});

// 🔄 Background sync function
function doBackgroundSync() {
    return new Promise((resolve, reject) => {
        // Simulate background data sync
        console.log('📊 Performing background sync...');
        
        // Get stored predictions to sync
        const predictions = JSON.parse(localStorage.getItem('plantPredictions') || '[]');
        
        if (predictions.length > 0) {
            console.log(`📤 Syncing ${predictions.length} predictions...`);
            
            // Simulate API call
            setTimeout(() => {
                console.log('✅ Background sync completed!');
                resolve();
            }, 2000);
        } else {
            console.log('📭 No data to sync');
            resolve();
        }
    });
}

// 🎭 Message Event - Communication with main thread
self.addEventListener('message', event => {
    console.log('💬 Message received:', event.data);
    
    switch (event.data.type) {
        case 'SKIP_WAITING':
            self.skipWaiting();
            break;
            
        case 'GET_VERSION':
            event.ports[0].postMessage({
                version: CACHE_NAME,
                staticCache: STATIC_CACHE,
                dynamicCache: DYNAMIC_CACHE
            });
            break;
            
        case 'CLEAR_CACHE':
            event.waitUntil(
                caches.keys().then(cacheNames => {
                    return Promise.all(
                        cacheNames.map(cache => caches.delete(cache))
                    );
                }).then(() => {
                    event.ports[0].postMessage({ success: true });
                })
            );
            break;
            
        default:
            console.log('🤷 Unknown message type:', event.data.type);
    }
});

// 🏠 App Badge API Support
self.addEventListener('appbadgechange', event => {
    console.log('🏠 App badge changed:', event.badge);
});

// 📱 Periodic Background Sync (experimental)
self.addEventListener('periodicsync', event => {
    if (event.tag === 'content-sync') {
        event.waitUntil(doPeriodicSync());
    }
});

function doPeriodicSync() {
    return new Promise((resolve) => {
        console.log('⏰ Periodic sync triggered');
        
        // Update model data or sync predictions
        const lastSync = localStorage.getItem('lastPeriodicSync');
        const now = Date.now();
        
        if (!lastSync || (now - parseInt(lastSync)) > 24 * 60 * 60 * 1000) {
            console.log('🔄 Performing periodic data update...');
            localStorage.setItem('lastPeriodicSync', now.toString());
        }
        
        resolve();
    });
}

// 🎯 Installation prompt
self.addEventListener('beforeinstallprompt', event => {
    console.log('📱 Installation prompt available');
    event.preventDefault();
    return event;
});

console.log('🔧 Service Worker loaded successfully!');
console.log('📦 Cache version:', CACHE_NAME);
console.log('🌟 Plant Classifier PWA Ready!');